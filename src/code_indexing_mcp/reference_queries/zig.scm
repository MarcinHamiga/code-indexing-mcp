; Structural reference captures for Zig. Roots marked @reference.* are
; forwarded to _zig_records, which re-dispatches on node.type;
; sub-captures are decorative.

; --- imports: @import("path") builtins (handler reads the string) -----------

(BUILTINIDENTIFIER) @reference.zig_import

; --- calls: member calls through field chains -------------------------------

(FieldOrFnCall
  (FnCallArguments) @arguments) @reference.call

; --- non-call member access through field chains ----------------------------

(SuffixExpr) @reference.member_access

; --- type-bearing declarations (handler descends the type subtree) -----------

(VarDecl) @reference.var_decl
(FnProto) @reference.fn_proto
(ParamDecl) @reference.param_decl
(ContainerField) @reference.container_field

; --- declaration parameters --------------------------------------------------

(FnProto
  (ParamDeclList) @declaration.parameters)

; --- exports: `pub` top-level declarations ---------------------------------

(source_file (Decl) @reference.export)

; --- identifier fallback (bindings excluded in _identifier_record) ------------

(IDENTIFIER) @reference.identifier
