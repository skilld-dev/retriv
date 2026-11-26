import type { BaseDriverConfig, Document, IndexOptions, SearchOptions, SearchProvider, SearchResult } from '../types'
import { Redis, s } from '@upstash/redis'
import { matchesFilter } from '../filter'
import { extractSnippet } from '../utils/extract-snippet'

export type UpstashRedisSearchMode = 'hybrid' | 'semantic' | 'fulltext'

export interface UpstashRedisConfig extends BaseDriverConfig {
  /** Upstash Redis REST URL (defaults to `Redis.fromEnv()` when url, token and redis are omitted) */
  url?: string
  /** Upstash Redis REST token */
  token?: string
  /** Existing `@upstash/redis` client, instead of url/token */
  redis?: Redis
  /** Redis Search index name (default: `retriv`) */
  index?: string
  /** Key prefix for stored documents (default: `<index>:`) */
  prefix?: string
  /**
   * - `hybrid` (default): BM25 and semantic search fused with RRF, server-side
   * - `semantic`: semantic search only
   * - `fulltext`: BM25 only
   */
  mode?: UpstashRedisSearchMode
}

interface StoredDocument {
  content: string
  metadata?: Record<string, any>
}

/** `SEARCH.QUERY` accepts LIMIT 1..1000 */
const MAX_LIMIT = 1000
const RRF_RANK_CONSTANT = 60
/** Best possible RRF score: rank 1 in both retrievers */
const RRF_MAX_SCORE = 2 / (RRF_RANK_CONSTANT + 1)
const WRITE_BATCH_SIZE = 100

const schema = s.object({ content: s.hybrid() })

/**
 * Create an Upstash Redis Search provider
 * Hybrid: BM25 + semantic search in one Upstash Redis database.
 * Embeddings are created server-side (no local embedding needed).
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
export async function upstashRedis(config: UpstashRedisConfig = {}): Promise<SearchProvider> {
  const { url, token, mode = 'hybrid' } = config

  if (!config.redis && (url || token) && !(url && token)) {
    throw new Error('[upstash-redis] url and token must be provided together')
  }

  const redis = config.redis ?? (url && token ? new Redis({ url, token }) : Redis.fromEnv())
  const indexName = config.index || 'retriv'
  const prefix = config.prefix || `${indexName}:`

  // Idempotent: keeps an identical existing index
  const index = await redis.search.createIndex({
    name: indexName,
    dataType: 'json',
    prefix,
    schema,
    existsOk: true,
  })

  async function scanKeys(): Promise<string[]> {
    const keys: string[] = []
    let cursor = '0'
    do {
      const [next, page] = await redis.scan(cursor, { match: `${prefix}*`, count: 1000, type: 'json' })
      keys.push(...page)
      cursor = String(next)
    } while (cursor !== '0')
    return keys
  }

  function buildFilter(query: string) {
    const ann = { $ann: { field: 'content' as const, query } }
    if (mode === 'semantic')
      return ann
    if (mode === 'fulltext')
      return { content: query }
    return { $rrf: { retrievers: [{ content: query }, ann], rankConstant: RRF_RANK_CONSTANT } }
  }

  function normalizeScore(score: number): number {
    if (mode === 'hybrid')
      score = score / RRF_MAX_SCORE
    else if (mode === 'fulltext')
      score = score / (1 + score) // BM25 is unbounded
    return Math.max(0, Math.min(1, score))
  }

  return {
    async index(docs: Document[], options?: IndexOptions) {
      if (docs.length === 0) {
        return { count: 0 }
      }

      const onProgress = options?.onProgress
      onProgress?.({ phase: 'storing', current: 0, total: docs.length })

      for (let i = 0; i < docs.length; i += WRITE_BATCH_SIZE) {
        const batch = docs.slice(i, i + WRITE_BATCH_SIZE)
        const pipeline = redis.pipeline()
        for (const doc of batch) {
          const stored: Record<string, unknown> = { content: doc.content }
          if (doc.metadata)
            stored.metadata = doc.metadata
          pipeline.json.set(`${prefix}${doc.id}`, '$', stored)
        }
        await pipeline.exec()
        onProgress?.({ phase: 'storing', current: i + batch.length, total: docs.length })
      }

      // Writes are indexed asynchronously; make them searchable before returning
      await index.waitIndexing()

      return { count: docs.length }
    },

    async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
      const { limit = 10, returnContent = false, returnMetadata = true, filter } = options
      if (!query.trim())
        return []

      const fetchLimit = Math.min(MAX_LIMIT, filter ? limit * 4 : limit)
      const results = await index.query({
        filter: buildFilter(query),
        limit: fetchLimit,
      })

      let mapped = (results || []).map((r) => {
        const data = r.data as StoredDocument | undefined
        const result: SearchResult = {
          id: r.key.slice(prefix.length),
          score: normalizeScore(r.score),
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
      if (ids.length === 0)
        return { count: 0 }
      await redis.del(...ids.map(id => `${prefix}${id}`))
      await index.waitIndexing()
      return { count: ids.length }
    },

    async listIds() {
      const keys = await scanKeys()
      return keys.map(key => key.slice(prefix.length))
    },

    async clear() {
      // Deleting the documents removes them from the index; the index itself is kept
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
