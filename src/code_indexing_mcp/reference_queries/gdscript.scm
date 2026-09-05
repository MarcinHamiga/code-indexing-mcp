; Structural reference captures for GDScript. Roots marked @reference.* are
; forwarded to _gdscript_records, which re-dispatches on node.type;
; sub-captures are decorative. The grammar is fieldless outside definitions,
; so the handler works positionally for calls and member access.

; --- calls: plain (`add_child(x)`) and member (`node.add_child(x)`) -----------
; (`attribute_call` is the `node.method(...)` shape; the enclosing `attribute`
; is skipped by the handler so the call row owns the span.)

(call) @reference.call

(attribute_call) @reference.attribute_call

; --- non-call member access (`node.hp`, `$Path.hp`) -----------------------------

(attribute) @reference.member_access

; --- inheritance and path imports (`extends Node`, `extends "res://b.gd"`) ------

(extends_statement) @reference.extends

; --- writes to bare names (`hp = 1` rebinds; `var` owns the declaration) --------

(assignment) @reference.assignment

(augmented_assignment) @reference.assignment

; --- declaration parameters -----------------------------------------------------

(function_definition
  parameters: (parameters) @declaration.parameters)

; --- identifier fallback (bindings excluded in _identifier_record) ---------------
; (`name` nodes -- function, variable, signal, and class names -- are a
; distinct node type, so the fallback never sees a declaration spelling.)

(identifier) @reference.identifier
