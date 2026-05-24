import type { Embedding, EmbeddingConfig, EmbeddingProvider, ResolvedEmbedding } from '../types'
import { getModelDimensions, getModelMaxTokens } from './model-info'

export interface WorkersAiBinding {
  run: (model: string, input: Record<string, unknown>) => Promise<unknown>
}

export interface CloudflareWorkersAiEmbeddingOptions {
  /** Workers AI binding, usually `env.AI`. */
  ai: WorkersAiBinding
  /** Workers AI embedding model. */
  model?: string
  /** BGE pooling strategy. Keep this stable for a given Vectorize index. */
  pooling?: 'cls' | 'mean'
}

function readEmbeddings(response: unknown): Embedding[] {
  const data = (response as { data?: unknown }).data
  if (!Array.isArray(data))
    throw new Error('[cloudflare-workers-ai] embedding response missing data array')
  return data.map((item) => {
    if (Array.isArray(item))
      return item as number[]
    if (item instanceof Float32Array)
      return item
    throw new TypeError('[cloudflare-workers-ai] embedding response contains a non-vector item')
  })
}

/**
 * Cloudflare Workers AI embedding provider.
 *
 * Pair this with `retriv/db/cloudflare` to index/search a Vectorize binding
 * from a Worker without leaving Cloudflare:
 *
 * ```ts
 * const search = await cloudflare({
 *   binding: env.KEYWORD_VECTORS,
 *   embeddings: cloudflareWorkersAi({ ai: env.AI }),
 * })
 * ```
 */
export function cloudflareWorkersAi(options: CloudflareWorkersAiEmbeddingOptions): EmbeddingConfig {
  const { ai, model = '@cf/baai/bge-small-en-v1.5', pooling = 'cls' } = options
  let cached: ResolvedEmbedding | null = null

  return {
    async resolve() {
      if (cached)
        return cached

      let dimensions = getModelDimensions(model)
      if (!dimensions) {
        const [embedding] = readEmbeddings(await ai.run(model, { text: ['test'], pooling }))
        dimensions = embedding?.length
      }
      if (!dimensions)
        throw new Error(`[cloudflare-workers-ai] could not determine dimensions for ${model}`)

      const embedder: EmbeddingProvider = async (texts) => {
        if (texts.length === 0)
          return []
        return readEmbeddings(await ai.run(model, { text: texts, pooling }))
      }

      cached = { embedder, dimensions, maxTokens: getModelMaxTokens(model) }
      return cached
    },
  }
}

export default cloudflareWorkersAi
