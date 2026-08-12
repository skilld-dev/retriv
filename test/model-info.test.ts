import { describe, expect, it } from 'vitest'
import { DEFAULT_MODELS, getModelDimensions, getModelMaxTokens, resolveModelForPreset } from '../src/embeddings/model-info'

describe('transformers.js preset mapping', () => {
  const presets = [
    ['bge-small-en-v1.5', 'Xenova/bge-small-en-v1.5', 384],
    ['bge-base-en-v1.5', 'Xenova/bge-base-en-v1.5', 768],
    ['bge-large-en-v1.5', 'Xenova/bge-large-en-v1.5', 1024],
    ['bge-m3', 'Xenova/bge-m3', 1024],
    ['all-MiniLM-L6-v2', 'Xenova/all-MiniLM-L6-v2', 384],
  ] as const

  it.each(presets)('maps %s to a repo with published weights', (preset, repo, dims) => {
    expect(resolveModelForPreset(preset, 'transformers.js')).toBe(repo)
    expect(getModelDimensions(preset)).toBe(dims)
  })

  it('passes through fully-qualified repo ids untouched', () => {
    expect(resolveModelForPreset('Xenova/bge-base-en-v1.5', 'transformers.js'))
      .toBe('Xenova/bge-base-en-v1.5')
  })

  it('resolves dimensions and max tokens through repo prefixes', () => {
    expect(getModelDimensions('Xenova/bge-large-en-v1.5')).toBe(1024)
    expect(getModelMaxTokens('Xenova/bge-large-en-v1.5')).toBe(512)
  })

  it('keeps the transformers.js default consistent with its mapping', () => {
    const fallback = DEFAULT_MODELS['transformers.js']
    expect(getModelDimensions(fallback.model)).toBe(fallback.dimensions)
  })
})
