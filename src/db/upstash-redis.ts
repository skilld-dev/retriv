import type { BaseDriverConfig, Document, IndexOptions, SearchOptions, SearchProvider, SearchResult } from '../types'
import { Redis, s } from '@upstash/redis'
import { matchesFilter } from '../filter'
import { extractSnippet } from '../utils/extract-snippet'

export interface UpstashRedisConfig extends BaseDriverConfig {
  /** Upstash Redis REST URL */
  url: string
  /** Upstash Redis REST token */
  token: string
  /** Redis Search index name, also the key prefix (default: `retriv`) */
  index?: string
}

interface StoredDocument {
  content: string
  metadata?: Record<string, any>
}

const WRITE_BATCH_SIZE = 100

/**
 * Create an Upstash Redis Search provider
 * Hybrid: BM25 + semantic search fused server-side (no local embedding needed)
 *
 * @example
 * ```ts
 * import { upstashRedis } from 'retriv/db/upstash-redis'
 *
 * const db = await upstashRedis({
 *   url: process.env.UPSTASH_REDIS_REST_URL,
 *   token: process.env.UPSTASH_REDIS_REST_TOKEN,
 * })
 * ```
 */
export async function upstashRedis(config: UpstashRedisConfig): Promise<SearchProvider> {
  const { url, token } = config

  if (!url) {
    throw new Error('[upstash-redis] url is required')
  }

  if (!token) {
    throw new Error('[upstash-redis] token is required')
  }

  const redis = new Redis({ url, token })
  const indexName = config.index || 'retriv'
  const prefix = `${indexName}:`

  const index = await redis.search.createIndex({
    name: indexName,
    dataType: 'json',
    prefix,
    schema: s.object({ content: s.hybrid() }),
    existsOk: true,
  })

  async function scanKeys(): Promise<string[]> {
    const keys: string[] = []
    let cursor = '0'
    do {
      const [next, page] = await redis.scan(cursor, { match: `${prefix}*`, count: 1000 })
      keys.push(...page)
      cursor = String(next)
    } while (cursor !== '0')
    return keys
  }

  return {
    async index(docs: Document[], options?: IndexOptions) {
      if (docs.length === 0) {
        return { count: 0 }
      }

      const onProgress = options?.onProgress
      onProgress?.({ phase: 'storing', current: 0, total: docs.length })

      for (let i = 0; i < docs.length; i += WRITE_BATCH_SIZE) {
        const pipeline = redis.pipeline()
        for (const doc of docs.slice(i, i + WRITE_BATCH_SIZE))
          pipeline.json.set(`${prefix}${doc.id}`, '$', { content: doc.content, metadata: doc.metadata })
        await pipeline.exec()
      }

      // Writes are indexed asynchronously; make them searchable before returning
      await index.waitIndexing()

      onProgress?.({ phase: 'storing', current: docs.length, total: docs.length })

      return { count: docs.length }
    },

    async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
      const { limit = 10, returnContent = false, returnMetadata = true, filter } = options
      if (!query.trim())
        return []

      const fetchLimit = filter ? limit * 4 : limit

      const results = await index.query({
        filter: { $rrf: { retrievers: [{ content: query }, { $ann: { field: 'content', query } }] } },
        limit: fetchLimit,
      })

      let mapped = (results || []).map((r) => {
        const data = r.data as StoredDocument | undefined
        const result: SearchResult = {
          id: r.key.slice(prefix.length),
          score: Math.max(0, Math.min(1, r.score)),
        }

        if (returnContent && data?.content) {
          const { snippet, highlights } = extractSnippet(data.content, query)
          result.content = snippet
          if (highlights.length)
            result._meta = { ...result._meta, highlights }
        }

        if (data?.metadata && Object.keys(data.metadata).length > 0)
          result.metadata = data.metadata

        return result
      })

      if (filter)
        mapped = mapped.filter(r => matchesFilter(filter, r.metadata))

      if (!returnMetadata)
        mapped.forEach((r) => { delete r.metadata })

      return mapped.slice(0, limit)
    },

    async remove(ids: string[]) {
      if (ids.length > 0) {
        await redis.del(...ids.map(id => `${prefix}${id}`))
        await index.waitIndexing()
      }
      return { count: ids.length }
    },

    async listIds() {
      const keys = await scanKeys()
      return keys.map(key => key.slice(prefix.length))
    },

    async clear() {
      const keys = await scanKeys()
      for (let i = 0; i < keys.length; i += 1000)
        await redis.del(...keys.slice(i, i + 1000))
      await index.waitIndexing()
    },

    async close() {
      // No-op for Upstash (HTTP)
    },
  }
}

export default upstashRedis
