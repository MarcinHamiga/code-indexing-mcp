; Structural reference captures for Kotlin. Roots marked @reference.* are
; forwarded to _kotlin_records, which re-dispatches on node.type;
; sub-captures are decorative.

; --- imports: one row per header (plain and wildcard) -----------------------

(import_header) @reference.import

; --- calls: plain identifier calls and navigation calls --------------------

(call_expression) @reference.call

; --- non-call member access through navigation expressions ------------------

(navigation_expression) @reference.member_access

; --- declaration parameters -------------------------------------------------

(function_declaration
  (function_value_parameters) @declaration.parameters)

; --- identifier fallback (bindings excluded in _identifier_record) -----------

(simple_identifier) @reference.identifier
(type_identifier) @reference.identifier
