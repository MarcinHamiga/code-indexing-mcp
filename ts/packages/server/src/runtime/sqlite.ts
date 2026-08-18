/** Bun's synchronous SQLite API behind the runtime-only boundary. */

import { Database } from "bun:sqlite";

export type SQLiteValue = string | number | bigint | boolean | Uint8Array | null;
export type SQLiteRow = Record<string, SQLiteValue>;

export interface SQLiteStatement {
  all(...parameters: SQLiteValue[]): SQLiteRow[];
  get(...parameters: SQLiteValue[]): SQLiteRow | null;
  run(...parameters: SQLiteValue[]): { changes: number; lastInsertRowid: number | bigint };
}

export interface SQLiteDatabase {
  exec(sql: string): void;
  query(sql: string): SQLiteStatement;
  close(): void;
}

class BunSQLiteDatabase implements SQLiteDatabase {
  readonly #database: Database;

  constructor(filename: string) {
    this.#database = new Database(filename, { create: true, readwrite: true });
  }

  exec(sql: string): void {
    this.#database.exec(sql);
  }

  query(sql: string): SQLiteStatement {
    return this.#database.query(sql) as unknown as SQLiteStatement;
  }

  close(): void {
    this.#database.close();
  }
}

/** Open a persistent SQLite database using Bun's built-in driver. */
export function openSQLite(filename: string): SQLiteDatabase {
  return new BunSQLiteDatabase(filename);
}
