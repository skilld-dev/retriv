import { beforeEach, describe, expect, it, vi } from 'vitest'

const { pipelineMock } = vi.hoisted(() => ({ pipelineMock: vi.fn() }))

vi.mock('@huggingface/transformers', () => ({
  env: { cacheDir: undefined },
  pipeline: pipelineMock,
}))

const { transformersJs } = await import('../src/embeddings/transformers-js')

describe('transformersJs pipeline options', () => {
  beforeEach(() => {
    pipelineMock.mockReset()
    pipelineMock.mockResolvedValue(async () => ({ data: new Float32Array(0) }))
  })

  it('defaults to fp32 and leaves device unset', async () => {
    await transformersJs({ model: 'bge-small-en-v1.5' }).resolve()

    const [task, model, opts] = pipelineMock.mock.calls[0]!
    expect(task).toBe('feature-extraction')
    expect(model).toBe('Xenova/bge-small-en-v1.5')
    expect(opts).toEqual({ dtype: 'fp32' })
  })

  it('forwards device and dtype to the pipeline', async () => {
    await transformersJs({
      model: 'bge-base-en-v1.5',
      device: 'coreml',
      dtype: 'q8',
    }).resolve()

    const [, model, opts] = pipelineMock.mock.calls[0]!
    expect(model).toBe('Xenova/bge-base-en-v1.5')
    expect(opts).toMatchObject({ device: 'coreml', dtype: 'q8' })
  })

  it('forwards per-file device and dtype maps', async () => {
    const device = { 'model.onnx': 'webgpu' } as const
    const dtype = { 'model.onnx': 'q8' } as const

    await transformersJs({ model: 'bge-base-en-v1.5', device, dtype }).resolve()

    const [, , opts] = pipelineMock.mock.calls[0]!
    expect(opts).toMatchObject({ device, dtype })
  })

  it('forwards device without overriding the default dtype', async () => {
    await transformersJs({ model: 'bge-small-en-v1.5', device: 'webgpu' }).resolve()

    const [, , opts] = pipelineMock.mock.calls[0]!
    expect(opts).toEqual({ dtype: 'fp32', device: 'webgpu' })
  })

  it('resolves dimensions for the selected model', async () => {
    const resolved = await transformersJs({ model: 'bge-large-en-v1.5' }).resolve()
    expect(resolved.dimensions).toBe(1024)
  })

  it('probes dimensions for a model missing from the registry', async () => {
    pipelineMock.mockResolvedValue(async () => ({ data: new Float32Array(384) }))

    const resolved = await transformersJs({ model: 'some-org/unlisted-model' }).resolve()

    expect(resolved.dimensions).toBe(384)
  })

  it('prefers an explicit dimensions option over probing', async () => {
    const extractor = vi.fn(async () => ({ data: new Float32Array(384) }))
    pipelineMock.mockResolvedValue(extractor)

    const resolved = await transformersJs({ model: 'some-org/unlisted-model', dimensions: 512 }).resolve()

    expect(resolved.dimensions).toBe(512)
    expect(extractor).not.toHaveBeenCalled()
  })
})
