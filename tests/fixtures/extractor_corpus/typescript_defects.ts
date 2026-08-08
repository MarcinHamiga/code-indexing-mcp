// Pins current (buggy) extraction for backlog defects E3, E7, E8, E10.
// E1 and E2 were fixed by Task 2.1 (name-descent helper for class_heritage,
// generic_type, union_type, intersection_type); the constructs below stay for
// snapshot coverage of the now-correct output.
//
// Task 0.1 freezes the wrong output rather than fixing it -- see the
// hardening plan, Phase 0.

export * from "./x"; // E3: barrel re-export produces no reference at all
export * as ns from "./x"; // E3: namespaced barrel re-export loses the module path

interface Comparable<T> {
  compareTo(other: T): number;
}

class Base {
  run(): number {
    return 1;
  }
}

class Derived<T> extends Base implements Comparable<T> {
  // E1 (fixed, Task 2.1): class_heritage now descends to `Base` and
  // `Comparable` plus a type_use for `T`, instead of the raw clause text.
  compareTo(other: T): number {
    return 0;
  }
}

type Union = Base | Derived<Base>;
type Intersection = Base & Comparable<Base>;

function identify(value: Union): Intersection {
  // E2 (fixed, Task 2.1): generic/union/intersection inner names now surface
  // as their own type_use rows.
  return value as unknown as Intersection;
}

abstract class Worker {
  // E10: abstract_class_declaration produces no declaration at all, so members
  // below lose their `Worker.` qualification.
  abstract run(): number; // E10: abstract_method_signature produces no declaration

  handle = (a: number, b: number): number => a + b; // E10: class-field arrow, no declaration
}

function describe({
  title,
  subtitle,
  footnote,
}: {
  title: string;
  subtitle: string;
  footnote: string;
}) {
  // E7: multi-key destructured parameter collapses to one fabricated name.
  return `${title} ${subtitle} ${footnote}`;
}

function bind(handler: (event: Event) => void, retries: number) {
  // E8: "=" inside "=>" misfires the default-detection heuristic, so `handler`
  // is read as optional and a missing-argument call never flags it.
  return retries;
}
