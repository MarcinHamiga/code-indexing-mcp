; Only the root <template> is a definition: nested <template v-if>/<template #slot>
; elements are markup, and as container captures they would truncate the root chunk.
(document
  (template_element
    (start_tag
      (tag_name) @name)) @definition.object)

(script_element
  (start_tag
    (tag_name) @name)) @definition.object

(style_element
  (start_tag
    (tag_name) @name)) @definition.object
