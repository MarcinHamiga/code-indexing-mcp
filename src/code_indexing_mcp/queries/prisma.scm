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

; Block attributes (@@unique, @@index, @@map, @@schema) as members of their block.
(model_multi_attribute
  (attribute_specifier
    (identifier) @name)) @definition.property
