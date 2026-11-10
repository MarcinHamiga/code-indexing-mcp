(preproc_def
  name: (identifier) @name) @definition.constant

; A struct is a definition only where it has a body: `static struct node head;`
; and `struct node *next;` are uses of the type, not declarations of it.
(struct_specifier
  name: (type_identifier) @name
  body: (field_declaration_list)) @definition.struct

(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name)) @definition.function

; Pointer-returning functions (`char *dup_name(...)`, `char **argv_copy(...)`)
; wrap the function_declarator in one pointer_declarator per `*`.
(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @name))) @definition.function

(function_definition
  declarator: (pointer_declarator
    declarator: (pointer_declarator
      declarator: (function_declarator
        declarator: (identifier) @name)))) @definition.function
