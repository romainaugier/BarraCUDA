/* terrs.c -- error handling
 * Making sure the compiler fails gracefully, not dramatically. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- errors: syntax ---- */

static void err_synt(void)
{
    int rc = th_run(BC_BIN " --amdgpu-bin tests/test_errors.cu -o err_test.hsaco",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    CHECK(strstr(obuf, "error") != NULL);
    remove("err_test.hsaco");
    PASS();
}
TH_REG("errors", err_synt)

/* ---- errors: missing file ---- */

static void err_miss(void)
{
    int rc = th_run(BC_BIN " --amdgpu-bin nonexistent_file_42.cu -o err_test.hsaco",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    remove("err_test.hsaco");
    PASS();
}
TH_REG("errors", err_miss)

/* ---- errors: bad output directory ---- */
/* /dev/null is a file, not a directory. You can't mkdir inside it.
 * Works on every Unix. On Windows we use NUL, same idea. */

static void err_odir(void)
{
    char cmd[TH_BUFSZ];
#ifdef _WIN32
    snprintf(cmd, TH_BUFSZ,
             BC_BIN " --amdgpu-bin tests/vector_add.cu "
             "-o NUL\\impossible\\out.hsaco");
#else
    snprintf(cmd, TH_BUFSZ,
             BC_BIN " --amdgpu-bin tests/vector_add.cu "
             "-o /dev/null/impossible/out.hsaco");
#endif
    int rc = th_run(cmd, obuf, TH_BUFSZ);
    (void)rc;
    CHECK(strstr(obuf, "cannot open") != NULL ||
          strstr(obuf, "error") != NULL);
    PASS();
}
TH_REG("errors", err_odir)
