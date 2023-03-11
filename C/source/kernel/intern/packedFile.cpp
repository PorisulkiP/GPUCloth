#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <io.h>

#include "MEM_guardedalloc.cuh"
#include <string.h>

#include "ID.h"
#include "DNA_image_types.h"
#include "DNA_packedFile_types.h"
#include "DNA_sound_types.h"
#include "DNA_vfont_types.h"
#include "DNA_volume_types.h"

#include "BLI_blenlib.h"
#include "utildefines.h"

#include "BKE_font.h"
#include "BKE_image.h"
#include "BKE_main.h"
#include "BKE_packedFile.h"
#include "BKE_report.h"
#include "BKE_sound.h"
#include "BKE_volume.h"

int BKE_packedfile_seek(PackedFile *pf, int offset, int whence)
{
  int oldseek = -1, seek = 0;

  if (pf) {
    oldseek = pf->seek;
    switch (whence) {
      case SEEK_CUR:
        seek = oldseek + offset;
        break;
      case SEEK_END:
        seek = pf->size + offset;
        break;
      case SEEK_SET:
        seek = offset;
        break;
      default:
        oldseek = -1;
        break;
    }
    if (seek < 0) {
      seek = 0;
    }
    else if (seek > pf->size) {
      seek = pf->size;
    }
    pf->seek = seek;
  }

  return oldseek;
}

void BKE_packedfile_rewind(PackedFile *pf)
{
  BKE_packedfile_seek(pf, 0, SEEK_SET);
}

int BKE_packedfile_read(PackedFile *pf, void *data, int size)
{
  if ((pf != NULL) && (size >= 0) && (data != NULL)) {
    if (size + pf->seek > pf->size) {
      size = pf->size - pf->seek;
    }

    if (size > 0) {
      memcpy(data, ((char *)pf->data) + pf->seek, size);
    }
    else {
      size = 0;
    }

    pf->seek += size;
  }
  else {
    size = -1;
  }

  return size;
}

void BKE_packedfile_free(PackedFile *pf)
{
  if (pf) {
    BLI_assert(pf->data != NULL);

    MEM_SAFE_FREE(pf->data);
    MEM_freeN(pf);
  }
  else {
    printf("%s: Trying to free a NULL pointer\n", __func__);
  }
}

PackedFile *BKE_packedfile_duplicate(const PackedFile *pf_src)
{
  BLI_assert(pf_src != NULL);
  BLI_assert(pf_src->data != NULL);

  PackedFile *pf_dst;

  pf_dst = MEM_dupallocN(pf_src);
  pf_dst->data = MEM_dupallocN(pf_src->data);

  return pf_dst;
}

PackedFile *BKE_packedfile_new_from_memory(void *mem, int memlen)
{
  BLI_assert(mem != NULL);

  PackedFile *pf = MEM_callocN(sizeof(*pf), "PackedFile");
  pf->data = mem;
  pf->size = memlen;

  return pf;
}

PackedFile *BKE_packedfile_new(ReportList *reports, const char *filename, const char *basepath)
{
  PackedFile *pf = NULL;
  int file, filelen;
  char name[FILE_MAX];
  void *data;

  /* render result has no filename and can be ignored
   * any other files with no name can be ignored too */
  if (filename[0] == '\0') {
    return pf;
  }

  // XXX waitcursor(1);

  /* convert relative filenames to absolute filenames */

  BLI_strncpy(name, filename, sizeof(name));
  BLI_path_abs(name, basepath);

  /* open the file
   * and create a PackedFile structure */

  file = BLI_open(name, O_BINARY | O_RDONLY, 0);
  if (file == -1) {
    BKE_reportf(reports, RPT_ERROR, "Unable to pack file, source path '%s' not found", name);
  }
  else {
    filelen = BLI_file_descriptor_size(file);

    if (filelen == 0) {
      /* MEM_mallocN complains about MEM_mallocN(0, "bla");
       * we don't care.... */
      data = MEM_mallocN(1, "packFile");
    }
    else {
      data = MEM_mallocN(filelen, "packFile");
    }
    if (read(file, data, filelen) == filelen) {
      pf = BKE_packedfile_new_from_memory(data, filelen);
    }
    else {
      MEM_freeN(data);
    }

    close(file);
  }

  // XXX waitcursor(0);

  return pf;
}

int BKE_packedfile_write_to_file(ReportList *reports,
                                 const char *ref_file_name,
                                 const char *filename,
                                 PackedFile *pf,
                                 const bool guimode)
{
  int file, number;
  int ret_value = RET_OK;
  bool remove_tmp = false;
  char name[FILE_MAX];
  char tempname[FILE_MAX];
  /*      void *data; */

  if (guimode) {
  }  // XXX  waitcursor(1);

  BLI_strncpy(name, filename, sizeof(name));
  BLI_path_abs(name, ref_file_name);

  if (BLI_exists(name)) {
    for (number = 1; number <= 999; number++) {
      BLI_snprintf(tempname, sizeof(tempname), "%s.%03d_", name, number);
      if (!BLI_exists(tempname)) {
        if (BLI_copy(name, tempname) == RET_OK) {
          remove_tmp = true;
        }
        break;
      }
    }
  }

  /* make sure the path to the file exists... */
  BLI_make_existing_file(name);

  file = BLI_open(name, O_BINARY + O_WRONLY + O_CREAT + O_TRUNC, 0666);
  if (file == -1) {
    BKE_reportf(reports, RPT_ERROR, "Error creating file '%s'", name);
    ret_value = RET_ERROR;
  }
  else {
    if (write(file, pf->data, pf->size) != pf->size) {
      BKE_reportf(reports, RPT_ERROR, "Error writing file '%s'", name);
      ret_value = RET_ERROR;
    }
    else {
      BKE_reportf(reports, RPT_INFO, "Saved packed file to: %s", name);
    }

    close(file);
  }

  if (remove_tmp) {
    if (ret_value == RET_ERROR) {
      if (BLI_rename(tempname, name) != 0) {
        BKE_reportf(reports,
                    RPT_ERROR,
                    "Error restoring temp file (check files '%s' '%s')",
                    tempname,
                    name);
      }
    }
    else {
      if (BLI_delete(tempname, false, false) != 0) {
        BKE_reportf(reports, RPT_ERROR, "Error deleting '%s' (ignored)", tempname);
      }
    }
  }

  if (guimode) {
  }  // XXX waitcursor(0);

  return ret_value;
}

/**
 * This function compares a packed file to a 'real' file.
 * It returns an integer indicating if:
 *
 * - PF_EQUAL:     the packed file and original file are identical
 * - PF_DIFFERENT: the packed file and original file differ
 * - PF_NOFILE:    the original file doesn't exist
 */
enum ePF_FileCompare BKE_packedfile_compare_to_file(const char *ref_file_name,
                                                    const char *filename,
                                                    PackedFile *pf)
{
  BLI_stat_t st;
  enum ePF_FileCompare ret_val;
  char buf[4096];
  char name[FILE_MAX];

  BLI_strncpy(name, filename, sizeof(name));
  BLI_path_abs(name, ref_file_name);

  if (BLI_stat(name, &st) == -1) {
    ret_val = PF_CMP_NOFILE;
  }
  else if (st.st_size != pf->size) {
    ret_val = PF_CMP_DIFFERS;
  }
  else {
    /* we'll have to compare the two... */

    const int file = BLI_open(name, O_BINARY | O_RDONLY, 0);
    if (file == -1) {
      ret_val = PF_CMP_NOFILE;
    }
    else {
      ret_val = PF_CMP_EQUAL;

      for (int i = 0; i < pf->size; i += sizeof(buf)) {
        int len = pf->size - i;
        if (len > sizeof(buf)) {
          len = sizeof(buf);
        }

        if (read(file, buf, len) != len) {
          /* read error ... */
          ret_val = PF_CMP_DIFFERS;
          break;
        }

        if (memcmp(buf, ((char *)pf->data) + i, len) != 0) {
          ret_val = PF_CMP_DIFFERS;
          break;
        }
      }

      close(file);
    }
  }

  return ret_val;
}

/**
 * #BKE_packedfile_unpack_to_file() looks at the existing files (abs_name, local_name)
 * and a packed file.
 *
 * It returns a char *to the existing file name / new file name or NULL when
 * there was an error or when the user decides to cancel the operation.
 *
 * \warning 'abs_name' may be relative still! (use a "//" prefix)
 * be sure to run #BLI_path_abs on it first.
 */
char *BKE_packedfile_unpack_to_file(ReportList *reports,
                                    const char *ref_file_name,
                                    const char *abs_name,
                                    const char *local_name,
                                    PackedFile *pf,
                                    enum ePF_FileStatus how)
{
  char *newname = NULL;
  const char *temp = NULL;

  if (pf != NULL) {
    switch (how) {
      case PF_KEEP:
        break;
      case PF_REMOVE:
        temp = abs_name;
        break;
      case PF_USE_LOCAL: {
        char temp_abs[FILE_MAX];

        BLI_strncpy(temp_abs, local_name, sizeof(temp_abs));
        BLI_path_abs(temp_abs, ref_file_name);

        /* if file exists use it */
        if (BLI_exists(temp_abs)) {
          temp = local_name;
          break;
        }
        /* else create it */
        ATTR_FALLTHROUGH;
      }
      case PF_WRITE_LOCAL:
        if (BKE_packedfile_write_to_file(reports, ref_file_name, local_name, pf, 1) == RET_OK) {
          temp = local_name;
        }
        break;
      case PF_USE_ORIGINAL: {
        char temp_abs[FILE_MAX];

        BLI_strncpy(temp_abs, abs_name, sizeof(temp_abs));
        BLI_path_abs(temp_abs, ref_file_name);

        /* if file exists use it */
        if (BLI_exists(temp_abs)) {
          BKE_reportf(reports, RPT_INFO, "Use existing file (instead of packed): %s", abs_name);
          temp = abs_name;
          break;
        }
        /* else create it */
        ATTR_FALLTHROUGH;
      }
      case PF_WRITE_ORIGINAL:
        if (BKE_packedfile_write_to_file(reports, ref_file_name, abs_name, pf, 1) == RET_OK) {
          temp = abs_name;
        }
        break;
      default:
        printf("%s: unknown return_value %u\n", __func__, how);
        break;
    }

    if (temp) {
      newname = BLI_strdup(temp);
    }
  }

  return newname;
}
