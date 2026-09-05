; Structural reference captures for Terraform (HCL). Roots marked @reference.*
; are forwarded to _terraform_records, which re-dispatches on node.type.
; HCL is fieldless, so the handler works positionally throughout.
;
; There is deliberately no identifier fallback: every value reference in HCL
; is a traversal (`variable_expr` plus `get_attr` chain), a function call, or
; an interpolation, all captured below. A bare identifier fallback would only
; read block types, attribute names, and object keys -- all bindings.

; --- traversals: `var.image_id`, `local.prefix`, `aws_instance.web.ip` -------

(expression) @reference.traversal

; --- interpolations inside strings: `"prefix-${var.env}"` --------------------

(template_interpolation) @reference.interpolation

; --- function calls: `max(1, var.n)` (type constructors skipped) --------------

(function_call) @reference.call

; --- blocks: only `module` sources become import edges -------------------------

(block) @reference.block
