; Structural reference captures for Lua. Roots marked @reference.* are forwarded
; to _lua_records, which re-dispatches on node.type; sub-captures are decorative.

; --- calls: prefix calls, method calls (`obj:method(x)`), and `require` ------

(function_call) @reference.call

; --- non-call member access through `.` and `:` -------------------------------

(dot_index_expression) @reference.member_access

(method_index_expression) @reference.member_access

; --- declaration parameters ---------------------------------------------------

(function_declaration
  parameters: (parameters) @declaration.parameters)

; --- identifier fallback (bindings excluded in _identifier_record) -------------

(identifier) @reference.identifier
