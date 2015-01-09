/*
execute with 

import numpy as np
import chain_forwards_backwards_native
print(chain_forwards_backwards_native)
import sys
print(sys.path)
chain_forwards_backwards_native.init_kappa(np.int(20))

will produce

(py3)sb358@fermat:~/crash$ python script.py
<module 'chain_forwards_backwards_native' from '/home/sb358/anaconda/envs/py3/lib/python3.4/site-packages/chain_forwards_backwards_native.cpython-34m.so'>
['/homes/mlghomes/sb358/crash', '/home/sb358/anaconda/envs/py3/lib/python34.zip', '/home/sb358/anaconda/envs/py3/lib/python3.4', '/home/sb358/anaconda/envs/py3/lib/python3.4/plat-linux', '/home/sb358/anaconda/envs/py3/lib/python3.4/lib-dynload', '/home/sb358/anaconda/envs/py3/lib/python3.4/site-packages', '/home/sb358/anaconda/envs/py3/lib/python3.4/site-packages/runipy-0.1.0-py3.4.egg', '/home/sb358/anaconda/envs/py3/lib/python3.4/site-packages/setuptools-3.6-py3.4.egg']
T_max: 833316904
T_max: 20
T_max != 20
T_max: 20
now set T_max=20 in C code
T_max == 20

*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdio.h>

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif

// Forward declarations

static PyObject * init_kappa(PyObject *self, PyObject *args);

static PyMethodDef module_functions[] = {
    {"init_kappa", init_kappa, METH_VARARGS,
     "init_kappa(T_max)\n\n"
     "creates an array for kappa to be used at every subsequent log-likelihood computation."  
     },
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

MOD_INIT(chain_forwards_backwards_native)
{
    PyObject *m;
    MOD_DEF(m, "chain_forwards_backwards_native", "C implementation of the forwards-backwards algorithm",
            module_functions)
    if (m == NULL)
        return MOD_ERROR_VAL;
 	import_array();  // Must be present for NumPy.
    return MOD_SUCCESS_VAL(m);

}

// Functions available to Python

static PyObject * init_kappa(PyObject *self, PyObject *args) {
    npy_intp T_max;
    printf("T_max: %u\n", T_max);
    if (!PyArg_ParseTuple(args, "i",
			&T_max)) {
		return NULL;
	}
    printf("T_max: %u\n", T_max);
    if (T_max == 20) 
        printf("T_max == 20\n");
    else 
        printf("T_max != 20\n");
    printf("T_max: %u\n", T_max);
    printf("now set T_max=20 in C code\n");
    T_max = 20;
    if (T_max == 20) 
        printf("T_max == 20\n");
    else 
        printf("T_max != 20\n");
	Py_INCREF(Py_None);
	return Py_None;
}
    

