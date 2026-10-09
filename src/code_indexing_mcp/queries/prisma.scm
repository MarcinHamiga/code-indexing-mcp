(model_block
  .
  (identifier) @name) @definition.record

(enum_block
  .
  (identifier) @name) @definition.enum

(model_field
  field_name: (identifier) @name) @definition.property

(key_value_block
  .
  (identifier) @name) @definition.object
