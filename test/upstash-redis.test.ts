import type { Redis } from '@upstash/redis'
import { describe, expect, it } from 'vitest'
import { upstashRedis } from '../src/db/upstash-redis'

/**
 * In-memory stand-in for the parts of `@upstash/redis` the driver uses.
 * It records the Redis Search calls and returns canned query results;
 * real ranking is covered by the e2e driver tests.
 */
function fakeRedis(queryResults: { key: string, score: number }[] = []) {
  const store = new Map<string, any>()
  const calls = { createIndex: [] as any[], query: [] as any[], waitIndexing: 0 }

  const index = {
    async waitIndexing() {
      calls.waitIndexing++
      return 1
    },
    async query(options: any) {
      calls.query.push(options)
      return queryResults.map(r => ({ ...r, data: store.get(r.key) }))
    },
  }

  const redis = {
    search: {
      async createIndex(params: any) {
        calls.createIndex.push(params)
        return index
      },
    },
    pipeline() {
      const queue: [string, any][] = []
      return {
        json: {
          set(key: string, _path: string, value: any) {
            queue.push([key, value])
          },
        },
        async exec() {
          for (const [key, value] of queue) store.set(key, value)
          return queue.map(() => 'OK')
        },
      }
    },
    async del(...keys: string[]) {
      let n = 0
      for (const key of keys) n += store.delete(key) ? 1 : 0
      return n
    },
    async scan(_cursor: string, { match }: { match: string }) {
      const prefix = match.slice(0, -1)
      return ['0', [...store.keys()].filter(k => k.startsWith(prefix))]
    },
  }

  return { redis: redis as unknown as Redis, store, calls }
}

describe('upstash-redis driver', () => {
  it('creates a hybrid JSON index once, idempotently', async () => {
    const { redis, calls } = fakeRedis()
    await upstashRedis({ redis, index: 'docs' })

    expect(calls.createIndex).toHaveLength(1)
    const params = calls.createIndex[0]
    expect(params).toMatchObject({ name: 'docs', dataType: 'json', prefix: 'docs:', existsOk: true })
    expect(params.schema).toEqual({ content: { type: 'HYBRID', noStem: undefined, from: undefined } })
  })

  it('stores documents as JSON and waits for indexing', async () => {
    const { redis, store, calls } = fakeRedis()
    const db = await upstashRedis({ redis })

    const res = await db.index([
      { id: 'a', content: 'alpha', metadata: { type: 'docs' } },
      { id: 'b', content: 'beta' },
    ])

    expect(res).toEqual({ count: 2 })
    expect(store.get('retriv:a')).toEqual({ content: 'alpha', metadata: { type: 'docs' } })
    expect(store.get('retriv:b')).toEqual({ content: 'beta' })
    expect(calls.waitIndexing).toBe(1)
  })

  it('runs a server-side RRF of full-text and semantic search by default', async () => {
    const { redis, calls } = fakeRedis()
    const db = await upstashRedis({ redis })
    await db.search('auth flow', { limit: 3 })

    expect(calls.query[0]).toEqual({
      filter: {
        $rrf: {
          retrievers: [{ content: 'auth flow' }, { $ann: { field: 'content', query: 'auth flow' } }],
          rankConstant: 60,
        },
      },
      limit: 3,
    })
  })

  it('supports semantic-only and fulltext-only modes', async () => {
    const semantic = fakeRedis()
    await (await upstashRedis({ redis: semantic.redis, mode: 'semantic' })).search('q')
    expect(semantic.calls.query[0].filter).toEqual({ $ann: { field: 'content', query: 'q' } })

    const fulltext = fakeRedis()
    await (await upstashRedis({ redis: fulltext.redis, mode: 'fulltext' })).search('q')
    expect(fulltext.calls.query[0].filter).toEqual({ content: 'q' })
  })

  it('normalizes scores to 0-1 and maps keys back to ids', async () => {
    const top = 2 / 61 // rank 1 in both retrievers
    const { redis } = fakeRedis([
      { key: 'retriv:a', score: top },
      { key: 'retriv:b', score: 1 / 61 },
    ])
    const db = await upstashRedis({ redis })
    await db.index([{ id: 'a', content: 'alpha' }, { id: 'b', content: 'beta' }])

    const results = await db.search('alpha')
    expect(results.map(r => r.id)).toEqual(['a', 'b'])
    expect(results[0]!.score).toBeCloseTo(1)
    expect(results[1]!.score).toBeCloseTo(0.5)
  })

  it('over-fetches and filters metadata client-side', async () => {
    const { redis, calls } = fakeRedis([
      { key: 'retriv:a', score: 0.03 },
      { key: 'retriv:b', score: 0.02 },
    ])
    const db = await upstashRedis({ redis })
    await db.index([
      { id: 'a', content: 'alpha', metadata: { type: 'docs' } },
      { id: 'b', content: 'beta', metadata: { type: 'code' } },
    ])

    const results = await db.search('x', { limit: 5, filter: { type: 'code' } })
    expect(calls.query[0].limit).toBe(20)
    expect(results.map(r => r.id)).toEqual(['b'])
    expect(results[0]!.metadata).toEqual({ type: 'code' })
  })

  it('returns content snippets and can drop metadata', async () => {
    const { redis } = fakeRedis([{ key: 'retriv:a', score: 0.03 }])
    const db = await upstashRedis({ redis })
    await db.index([{ id: 'a', content: 'line one\nalpha here\nline three', metadata: { type: 'docs' } }])

    const [result] = await db.search('alpha', { returnContent: true, returnMetadata: false })
    expect(result!.content).toContain('alpha here')
    expect(result!.metadata).toBeUndefined()
  })

  it('returns nothing for an empty query', async () => {
    const { redis, calls } = fakeRedis()
    const db = await upstashRedis({ redis })
    expect(await db.search('   ')).toEqual([])
    expect(calls.query).toHaveLength(0)
  })

  it('removes, lists and clears documents under its prefix', async () => {
    const { redis, store } = fakeRedis()
    store.set('other:x', { content: 'not ours' })
    const db = await upstashRedis({ redis })
    await db.index([{ id: 'a', content: 'alpha' }, { id: 'b#chunk-1', content: 'beta' }])

    expect((await db.listIds!()).sort()).toEqual(['a', 'b#chunk-1'])
    expect(await db.remove!(['a'])).toEqual({ count: 1 })
    expect(await db.listIds!()).toEqual(['b#chunk-1'])

    await db.clear!()
    expect(await db.listIds!()).toEqual([])
    expect(store.has('other:x')).toBe(true)
  })

  it('requires url and token together', async () => {
    await expect(upstashRedis({ url: 'https://example.upstash.io' })).rejects.toThrow('url and token')
  })
})
