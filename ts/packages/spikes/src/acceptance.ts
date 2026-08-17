/**
 * Port of `acceptance.py` -- the pure correctness metrics the accelerator
 * promotion gates already use.
 *
 * S3 scores the TypeScript embedder against Python-produced vectors with the
 * *same* metrics the project already trusts, rather than inventing a parity
 * measure for the migration. Its numbers are directly comparable to what
 * `test_accelerator_acceptance.py` asserts.
 *
 * **The canonical port now lives in `packages/server/src/acceptance.ts`** --
 * Phase 1 landed it, with `topKRankCorrelation` and a proper suite. This copy
 * stays deliberately: a Phase 0 spike is a record of an experiment, and one that
 * reached back into code written after it ran would no longer reproduce the
 * verdict in `docs/plans/2026-08-17-phase-0-spike-results.md`. Fix bugs in the
 * server module; change this one only to keep the spike runnable.
 *
 * Deliberately written over plain arrays: pulling in a linear-algebra
 * dependency to answer a Phase 0 question would prejudge a Phase 4 decision.
 */

export type Matrix = readonly (readonly number[])[];

function validate(matrix: Matrix, name: string): void {
  const rows = matrix.length;
  const width = matrix[0]?.length ?? 0;
  if (rows === 0 || width === 0) {
    throw new Error(`${name} must be a non-empty two-dimensional matrix`);
  }
  for (const row of matrix) {
    if (row.length !== width) throw new Error(`${name} is ragged`);
    for (const value of row) {
      if (!Number.isFinite(value)) throw new Error(`${name} contains non-finite values`);
    }
  }
}

/** L2-normalize each row, rejecting zero-length rows as `acceptance.py` does. */
export function normalized(matrix: Matrix, name: string): number[][] {
  validate(matrix, name);
  return matrix.map((row) => {
    const norm = Math.sqrt(row.reduce((total, value) => total + value * value, 0));
    if (norm <= 1e-12) throw new Error(`${name} contains a zero-length row`);
    return row.map((value) => value / norm);
  });
}

/** Cosine similarity for each corresponding pair of matrix rows. */
export function cosineRows(reference: Matrix, candidate: Matrix): number[] {
  const left = normalized(reference, "reference");
  const right = normalized(candidate, "candidate");
  if (left.length !== right.length || left[0]?.length !== right[0]?.length) {
    throw new Error(
      `reference and candidate shapes differ: ` +
        `${left.length}x${left[0]?.length} != ${right.length}x${right[0]?.length}`,
    );
  }
  return left.map((row, index) => {
    const other = right[index];
    if (other === undefined) throw new Error("row count mismatch");
    return row.reduce((total, value, column) => total + value * (other[column] ?? 0), 0);
  });
}

/** Indices of the k highest scores, ties broken by ascending index (stable). */
function topK(scores: readonly number[], k: number): number[] {
  return scores
    .map((score, index) => ({ score, index }))
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, k)
    .map((entry) => entry.index);
}

/** Mean top-k result overlap between two candidate vector matrices. */
export function topKOverlap(
  queries: Matrix,
  reference: Matrix,
  candidate: Matrix,
  k: number,
): number {
  const queryRows = normalized(queries, "queries");
  const referenceRows = normalized(reference, "reference");
  const candidateRows = normalized(candidate, "candidate");

  if (referenceRows.length !== candidateRows.length) {
    throw new Error(
      `reference and candidate shapes differ: ${referenceRows.length} != ${candidateRows.length}`,
    );
  }
  if (queryRows[0]?.length !== referenceRows[0]?.length) {
    throw new Error(
      `query dimension ${queryRows[0]?.length} does not match ` +
        `candidate dimension ${referenceRows[0]?.length}`,
    );
  }
  if (k < 1 || k > referenceRows.length) {
    throw new Error(`k must be from 1 to ${referenceRows.length}`);
  }

  const score = (query: readonly number[], rows: readonly (readonly number[])[]): number[] =>
    rows.map((row) =>
      row.reduce((total, value, column) => total + value * (query[column] ?? 0), 0),
    );

  let overlap = 0;
  for (const query of queryRows) {
    const referenceOrder = new Set(topK(score(query, referenceRows), k));
    for (const index of topK(score(query, candidateRows), k)) {
      if (referenceOrder.has(index)) overlap += 1;
    }
  }
  return overlap / (queryRows.length * k);
}

/**
 * Attention-mask mean pooling followed by L2 normalization.
 *
 * `direct_onnx.py::mean_pool_and_normalize`, which is the pooling the index
 * was built with. Getting this wrong produces vectors that look plausible and
 * rank differently, which is exactly the failure S3 exists to catch.
 */
export function meanPoolAndNormalize(
  hidden: Float32Array,
  attentionMask: readonly (readonly number[])[],
  shape: readonly [number, number, number],
): number[][] {
  const [batch, sequence, width] = shape;
  const pooled: number[][] = [];
  for (let row = 0; row < batch; row += 1) {
    const mask = attentionMask[row];
    if (mask === undefined) throw new Error(`attention mask is missing row ${row}`);
    const sums = new Array<number>(width).fill(0);
    let counted = 0;
    for (let token = 0; token < sequence; token += 1) {
      if ((mask[token] ?? 0) === 0) continue;
      counted += 1;
      const base = (row * sequence + token) * width;
      for (let column = 0; column < width; column += 1) {
        sums[column] = (sums[column] ?? 0) + (hidden[base + column] ?? 0);
      }
    }
    // numpy divides by the clamped mask sum; a fully-masked row would be a bug
    // upstream, so match the clamp rather than inventing a zero vector.
    const divisor = Math.max(counted, 1e-9);
    const averaged = sums.map((value) => value / divisor);
    const norm = Math.sqrt(averaged.reduce((total, value) => total + value * value, 0));
    pooled.push(averaged.map((value) => value / Math.max(norm, 1e-12)));
  }
  return pooled;
}
