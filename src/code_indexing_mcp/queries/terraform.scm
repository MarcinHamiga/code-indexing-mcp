(block
  (identifier)
  .
  (string_lit
    (quoted_template_start)
    (template_literal) @name
    (quoted_template_end))
  .
  (block_start)) @definition.object

(block
  (identifier)
  .
  (string_lit
    (quoted_template_start)
    (template_literal) @name
    (quoted_template_end))
  .
  (string_lit)) @definition.object
