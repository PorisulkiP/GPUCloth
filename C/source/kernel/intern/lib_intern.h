#pragma once

typedef void (*BKE_library_free_notifier_reference_cb)(const void*);
typedef void (*BKE_library_remap_editor_id_reference_cb)(struct ID*, struct ID*);

extern BKE_library_free_notifier_reference_cb free_notifier_reference_cb;
extern BKE_library_remap_editor_id_reference_cb remap_editor_id_reference_cb;
