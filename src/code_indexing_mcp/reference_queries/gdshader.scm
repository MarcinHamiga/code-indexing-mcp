; Structural reference captures for Godot shaders. Roots marked @reference.*
; are forwarded to _gdshader_records, which re-dispatches on node.type;
; sub-captures are decorative. Shaders have no module system, so there are no
; import or export rows -- only calls, member reads/writes, and type uses.

; --- calls: user functions (`brightness(x)`); builtin constructors ---------

(call_expression function: (_) @name arguments: (_) @arguments) @reference.call

; --- non-call member access (`COLOR.rgb`) ---------------------------------------

(field_expression) @reference.member_access

; --- struct field types (the member name itself is a binding) -------------------

(field_definition) @reference.declaration

; --- declaration parameters (fieldless: the handler takes the last name) --------

(function_definition
  parameters: (parameter_list) @declaration.parameters)

; --- identifier fallback (bindings excluded in _identifier_record) ---------------

(identifier) @reference.identifier
