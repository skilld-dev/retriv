import { describe, expect, it } from 'vitest'
import { codeChunker } from '../src/chunkers/typescript'

describe('codeChunker', () => {
  it('chunks a TypeScript file', async () => {
    const chunker = codeChunker()
    const code = `
import { ref } from 'vue'

export function useCounter() {
  const count = ref(0)
  function increment() {
    count.value++
  }
  return { count, increment }
}

export function useToggle(initial = false) {
  const value = ref(initial)
  function toggle() {
    value.value = !value.value
  }
  return { value, toggle }
}
`.trim()

    const chunks = await chunker(code, { id: 'composables.ts' })
    expect(chunks.length).toBeGreaterThanOrEqual(1)
    for (const chunk of chunks) {
      expect(chunk.text.length).toBeGreaterThan(0)
      if (chunk.lineRange) {
        expect(chunk.lineRange[0]).toBeTypeOf('number')
        expect(chunk.lineRange[1]).toBeTypeOf('number')
      }
    }
  })

  it('returns single chunk for small files', async () => {
    const chunker = codeChunker()
    const code = `export const x = 1`
    const chunks = await chunker(code, { id: 'small.ts' })
    expect(chunks.length).toBe(1)
    expect(chunks[0]!.text).toContain('export const x = 1')
  })

  it('respects maxChunkSize option', async () => {
    const chunker = codeChunker({ maxChunkSize: 100 })
    const code = `
export function longFunction() {
  const a = 1
  const b = 2
  const c = 3
  const d = 4
  const e = 5
  return a + b + c + d + e
}
`.trim()

    const chunks = await chunker(code, { id: 'long.ts' })
    // Should produce multiple chunks due to small maxChunkSize
    expect(chunks.length).toBeGreaterThanOrEqual(1)
  })

  it('keeps comments, keywords and trailing statements at chunk boundaries', async () => {
    const code = `export function alpha() {
  return 1
}

/** Beta doc comment */
export function beta() {
  return 2
}

export const GAMMA = 3

registerPlugin('trailing statement')
`
    const chunks = await codeChunker({ maxChunkSize: 40 })(code, { id: 'src/a.ts' })

    expect(chunks.map(c => c.text)).toEqual([
      'export function alpha() {\n  return 1\n}',
      '/** Beta doc comment */\nexport function beta() {\n  return 2\n}',
      'export const GAMMA = 3\n\nregisterPlugin(\'trailing statement\')',
    ])
    expect(chunks.map(c => c.lineRange)).toEqual([[1, 3], [5, 8], [10, 12]])
  })

  it('covers every line when declarations share one statement', async () => {
    const methods = Array.from({ length: 30 }, (_, i) => `  /** doc ${i} */\n  method${i}(raw) { return raw + '${i}' },`)
    const code = `module.exports = {\n${methods.join('\n')}\n}\nconsole.log('tail')\n`
    const chunks = await codeChunker({ maxChunkSize: 200 })(code, { id: 'bundle.js' })

    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks.map(c => c.text).join('').replace(/\s/g, '')).toBe(code.replace(/\s/g, ''))
  })
})
