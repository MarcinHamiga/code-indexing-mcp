/**
 * Pure correctness metrics shared by accelerator promotion gates.
 *
 * Written over plain arrays rather than a linear-algebra dependency: the
 * matrices here are a probe corpus, not a workload, and the whole module is on
 * no hot path. The Python original computes in float32 because that is what
 * numpy narrows an embedding matrix to; this one stays in float64, which can
 * only move a cosine by around 1e-7 -- five orders of magnitude below the
 * loosest gate any caller applies, and already confirmed harmless by the S3
 * embedding-parity spike.
 */

/** A non-empty, rectangular matrix of finite numbers. */
export type Matrix = readonly (readonly number[])[];

function validated(matrix: Matrix, name: string): Matrix {
  const width = matrix[0]?.length ?? 0;
  if (matrix.length === 0 || width === 0) {
    throw new Error(`${name} must be a non-empty two-dimensional matrix`);
  }
  for (const row of matrix) {
    if (row.length !== width) {
      throw new Error(`${name} must be a non-empty two-dimensional matrix`);
    }
    for (const value of row) {
      if (!Number.isFinite(value)) throw new Error(`${name} contains non-finite values`);
    }
  }
  return matrix;
}

/** L2-normalize each row, rejecting the zero-length rows a cosine cannot score. */
function normalized(matrix: Matrix, name: string): number[][] {
  return validated(matrix, name).map((row) => {
    const norm = Math.sqrt(row.reduce((total, value) => total + value * value, 0));
    if (norm <= 1e-12) throw new Error(`${name} contains a zero-length row`);
    return row.map((value) => value / norm);
  });
}

function shape(matrix: Matrix): string {
  return `(${matrix.length}, ${matrix[0]?.length ?? 0})`;
}

function dot(left: readonly number[], right: readonly number[]): number {
  return left.reduce((total, value, index) => total + value * (right[index] ?? 0), 0);
}

/** Return cosine similarity for each corresponding pair of matrix rows. */
export function cosineRows(reference: Matrix, candidate: Matrix): number[] {
  const left = normalized(reference, "reference");
  const right = normalized(candidate, "candidate");
  if (left.length !== right.length || (left[0]?.length ?? 0) !== (right[0]?.length ?? 0)) {
    throw new Error(`reference and candidate shapes differ: ${shape(left)} != ${shape(right)}`);
  }
  return left.map((row, index) => dot(row, right[index] ?? []));
}

/** Indices of the k highest scores, ties broken by ascending index. */
function topKIndices(scores: readonly number[], k: number): number[] {
  return scores
    .map((score, index) => ({ score, index }))
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, k)
    .map((entry) => entry.index);
}

/** Return mean top-k result overlap for two candidate vector matrices. */
export function topKOverlap(
  queries: Matrix,
  reference: Matrix,
  candidate: Matrix,
  { k }: { k: number },
): number {
  const queryRows = normalized(queries, "queries");
  const referenceRows = normalized(reference, "reference");
  const candidateRows = normalized(candidate, "candidate");
  if (
    referenceRows.length !== candidateRows.length ||
    (referenceRows[0]?.length ?? 0) !== (candidateRows[0]?.length ?? 0)
  ) {
    throw new Error(
      `reference and candidate shapes differ: ` +
        `${shape(referenceRows)} != ${shape(candidateRows)}`,
    );
  }
  if ((queryRows[0]?.length ?? 0) !== (referenceRows[0]?.length ?? 0)) {
    throw new Error(
      `query dimension ${queryRows[0]?.length ?? 0} does not match ` +
        `candidate dimension ${referenceRows[0]?.length ?? 0}`,
    );
  }
  if (!(k >= 1 && k <= referenceRows.length)) {
    throw new Error(`k must be from 1 to ${referenceRows.length}`);
  }

  let overlap = 0;
  for (const query of queryRows) {
    const referenceOrder = new Set(
      topKIndices(
        referenceRows.map((row) => dot(query, row)),
        k,
      ),
    );
    for (const index of topKIndices(
      candidateRows.map((row) => dot(query, row)),
      k,
    )) {
      if (referenceOrder.has(index)) overlap += 1;
    }
  }
  return overlap / (queryRows.length * k);
}

/**
 * Return the mean Kendall tau-b between paired top-k id rankings.
 *
 * Both rankings are top-k windows over one result set of unique ids, so an id
 * present in only one window counts as tied just past that window's end
 * (position `k`) in the other: losing a result outright is a tie among the lost
 * rather than an invented rank. 1.0 is an identical ranking and -1.0 a fully
 * reversed one over the same ids.
 */
export function topKRankCorrelation<T>(
  referenceOrders: readonly (readonly T[])[],
  candidateOrders: readonly (readonly T[])[],
): number {
  if (referenceOrders.length !== candidateOrders.length) {
    throw new Error(
      `ranking counts differ: ${referenceOrders.length} != ${candidateOrders.length}`,
    );
  }
  if (referenceOrders.length === 0) {
    throw new Error("at least one pair of rankings is required");
  }
  let total = 0;
  for (const [index, referenceOrder] of referenceOrders.entries()) {
    const candidateOrder = candidateOrders[index] ?? [];
    if (referenceOrder.length === 0 || candidateOrder.length === 0) {
      throw new Error("a top-k ranking must not be empty");
    }
    total += kendallTauB(referenceOrder, candidateOrder);
  }
  return total / referenceOrders.length;
}

function kendallTauB<T>(referenceOrder: readonly T[], candidateOrder: readonly T[]): number {
  const window = Math.max(referenceOrder.length, candidateOrder.length);
  const union = [...new Set([...referenceOrder, ...candidateOrder])];
  const rank = (order: readonly T[]): number[] => {
    const positions = new Map(order.map((item, position) => [item, position]));
    return union.map((item) => positions.get(item) ?? window);
  };
  const referenceRanks = rank(referenceOrder);
  const candidateRanks = rank(candidateOrder);

  let concordant = 0;
  let discordant = 0;
  let tiedReference = 0;
  let tiedCandidate = 0;
  for (let left = 0; left < union.length; left += 1) {
    for (let right = left + 1; right < union.length; right += 1) {
      const referenceSign = Math.sign(
        (referenceRanks[left] ?? window) - (referenceRanks[right] ?? window),
      );
      const candidateSign = Math.sign(
        (candidateRanks[left] ?? window) - (candidateRanks[right] ?? window),
      );
      if (referenceSign !== 0 && referenceSign === candidateSign) concordant += 1;
      if (referenceSign !== 0 && referenceSign === -candidateSign) discordant += 1;
      if (referenceSign === 0 && candidateSign !== 0) tiedReference += 1;
      if (candidateSign === 0 && referenceSign !== 0) tiedCandidate += 1;
    }
  }
  const denominator = Math.sqrt(
    (concordant + discordant + tiedReference) * (concordant + discordant + tiedCandidate),
  );
  // Both windows hold exactly the same single id: identical rankings.
  return denominator === 0 ? 1 : (concordant - discordant) / denominator;
}
