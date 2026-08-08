// Pins current (buggy) extraction for backlog defects E7, E10, E14 against
// the tsx grammar specifically -- reference_queries/tsx.scm is a separate
// compiled copy of typescript.scm, so a future fix that only touches one of
// the two mirrored files must show up as a diff here too.
// E1 was fixed by Task 2.1; the construct below stays for snapshot coverage
// of the now-correct output (and to prove tsx.scm got the same fix).
//
// Task 0.1 freezes the wrong output rather than fixing it -- see the
// hardening plan, Phase 0.

import type { ReactNode } from "react";

export interface Comparable<T> {
  compareTo(other: T): number;
}

class Base {
  run(): number {
    return 1;
  }
}

class Derived<T> extends Base implements Comparable<T> {
  // E1 (tsx copy, fixed Task 2.1): class_heritage now descends to `Base` and
  // `Comparable` plus a type_use for `T`.
  compareTo(other: T): number {
    return 0;
  }
}

abstract class Worker {
  // E10 (tsx copy): abstract_class_declaration produces no declaration.
  abstract run(): number;

  handle = (a: number, b: number): number => a + b; // E10 (tsx copy): class-field arrow
}

interface CardProps {
  title: string;
  subtitle: string;
  onSave: () => void;
}

function Widget({ label }: { label: string }) {
  return <span>{label}</span>;
}

export function Card({ title, subtitle, onSave }: CardProps) {
  // E7 (tsx copy): multi-key destructured parameter collapses to one name.
  return (
    <div className="card">
      <Widget label={title} />
      {/* E14: JSX element/component names never become a reference row. */}
      <span>{subtitle}</span>
      {onSave ? <button onClick={onSave}>Save</button> : null}
    </div>
  );
}
