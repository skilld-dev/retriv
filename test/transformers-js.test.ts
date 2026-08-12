import { pipeline } from '@huggingface/transformers'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { transformersJs } from '../src/embeddings/transformers-js'

vi.mock('@huggingface/transformers', () => ({
  env: {},
  pipeline: vi.fn(async () => vi.fn()),
}))

describe('transformersJs', () => {
  beforeEach(() => {
    vi.mocked(pipeline).mockClear()
  })

  it('runs the model on the selected device', async () => {
    await transformersJs({ model: 'bge-small-en-v1.5', device: 'webgpu' }).resolve()

    expect(pipeline).toHaveBeenCalledWith(
      'feature-extraction',
      'Xenova/bge-small-en-v1.5',
      expect.objectContaining({ device: 'webgpu' }),
    )
  })

  it('leaves device selection to transformers.js when omitted', async () => {
    await transformersJs({ model: 'bge-small-en-v1.5' }).resolve()

    expect(pipeline).toHaveBeenCalledWith(
      'feature-extraction',
      'Xenova/bge-small-en-v1.5',
      expect.not.objectContaining({ device: expect.anything() }),
    )
  })
})
