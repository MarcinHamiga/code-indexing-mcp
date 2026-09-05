; Structural reference captures for SQL. Roots marked @reference.* are forwarded
; to _sql_records, which re-dispatches on node.type. This is relational
; analysis, not name analysis: `relation` nodes are table reads (or writes
; under `update`), `field` nodes are column reads qualified by their table
; alias, `invocation` nodes are function calls, and `object_reference` nodes
; outside definitions are schema uses (drop/alter targets, index sources).
;
; There is deliberately no identifier fallback: every reference flows through
; one of the relational nodes above. Aliases, column definitions, and the
; names created by `create_*` statements are bindings, never reads.

; --- table relations in FROM/JOIN/UPDATE (handler picks read vs write) --------

(relation) @reference.relation

; --- columns: bare (`name`) and alias-qualified (`u.name`) ---------------------

(field) @reference.column

; --- function calls: `lower(name)`, `COUNT(*)` ----------------------------------

(invocation) @reference.call

; --- row mutations: `INSERT INTO users ...` -------------------------------------

(insert) @reference.dml

; --- remaining schema uses: drop/alter targets, indexed tables -----------------
; (Skipped under `relation`, `field`, `insert`, `column_definition`, and the
; `create_*` statements that define their name.)

(object_reference) @reference.object
