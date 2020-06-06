#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include <math.h>

typedef struct {
  PyObject_HEAD
  int memsize;
  int nlevels;
  int size;
  PyObject* data;
} SumTreeObject;

static void
SumTree_dealloc(SumTreeObject *self)
{
  Py_DECREF(self->data);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
SumTree_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    SumTreeObject *self;
    self = (SumTreeObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->nlevels = 0;
        self->size = 0;
        self->data = NULL;
    }
    return (PyObject *) self;
}

static int
SumTree_init(SumTreeObject *self, PyObject *args, PyObject *kwds)
{
    if (!PyArg_ParseTuple(args,"|i", &self->size))
        return -1;
    self->nlevels = (int)(ceil(log2(self->size)) + 1);
    self->memsize = (1 << self->nlevels) - 1;
    //self->data = calloc(sizeof(float) * self->memsize, 1);
    self->data = PyBytes_FromStringAndSize(NULL, sizeof(float) * self->memsize);
    float* data = PyBytes_AsString(self->data);
    for (int i=0;i<self->memsize;i++){
      data[i] = 0;
    }
    return 0;
}

static PyMemberDef SumTree_members[] = {
    {NULL}  /* Sentinel */
};

static PyObject *
SumTree_set(SumTreeObject *self, PyObject *args, PyObject *kwds)
{
  int idx;
  float p;
  if (!PyArg_ParseTuple(args, "if",  &idx, &p ))
    return NULL;
  int offset = (1 << (self->nlevels - 1));
  float* data = PyBytes_AsString(self->data);
  float delta = p - data[idx + offset];
  for (int i=self->nlevels - 1; i>=0; i--){
    data[idx + offset] += delta;
    idx >>= 1;
    offset >>= 1;
  }
  data[0] += delta;
  Py_RETURN_NONE;
}

static PyObject *
SumTree_get(SumTreeObject *self, PyObject *args, PyObject *kwds)
{
  int idx = 0;
  if (!PyArg_ParseTuple(args, "i", &idx ))
    return NULL;
  float* data = (float*)PyBytes_AsString(self->data);
  return PyFloat_FromDouble(data[idx + (1 << (self->nlevels - 1))]);
}

static PyObject *
SumTree_total(SumTreeObject *self, PyObject *args, PyObject *kwds)
{
  float* data = PyBytes_AsString(self->data);
  return PyFloat_FromDouble(data[0]);
}

static PyObject *
SumTree_sample(SumTreeObject *self, PyObject *args, PyObject *kwds)
{
  float q = 0;
  if (!PyArg_ParseTuple(args, "f", &q ))
    return NULL;
  float* data = (float*)PyBytes_AsString(self->data);
  q *= data[0];
  int s = 0;
  int level_offset = 1;
  for (int i=1; i<self->nlevels+1; i++){
    s *= 2;
    if (data[s + level_offset] < q && data[s + level_offset + 1] > 0){
      q -= data[s + level_offset];
      s += 1;
    }
    level_offset <<= 1;
  } 
  return PyLong_FromLong(s);
}

static PyObject *
SumTree_getstate(SumTreeObject *self, PyObject *args, PyObject *kwds)
{
  int header_size = sizeof(int) * 3;
  float* dataptr = PyBytes_AsString(self->data);
  PyObject* bytes = PyBytes_FromStringAndSize((const char*)&self->memsize, header_size);
  PyObject* data = PyBytes_FromStringAndSize((const char*)dataptr, self->memsize * sizeof(float));
  PyBytes_Concat(&bytes, data);
  Py_DECREF(data);
  return bytes;
}

static PyObject *
SumTree_setstate(SumTreeObject *self, PyObject *args, PyObject *kwds)
{
  PyObject* state;
  if (!PyArg_ParseTuple(args, "O", &state))
    return NULL;
  char* cstate = PyBytes_AsString(state);
  self->memsize = ((int*)cstate)[0];
  self->nlevels = ((int*)cstate)[1];
  self->size = ((int*)cstate)[2];
  self->data = PyBytes_FromStringAndSize(NULL, sizeof(float) * self->memsize);
  float* dataptr = PyBytes_AsString(self->data);
  //self->data = malloc(self->memsize * sizeof(float));
  float* dp = (float*)(cstate + sizeof(int) * 3);
  for (int i=0;i<self->memsize;i++){
    dataptr[i] = dp[i];
  }
  Py_RETURN_NONE;
}

static PyMethodDef SumTree_methods[] = {
    {"set", (PyCFunction) SumTree_set, METH_VARARGS,
     "set(idx, p)"
    },
    {"get", (PyCFunction) SumTree_get, METH_VARARGS,
     "get(idx)"
    },
    {"sample", (PyCFunction) SumTree_sample, METH_VARARGS,
     "sample(q)"
    },
    {"total", (PyCFunction) SumTree_total, METH_VARARGS,
     "total"
    },
    {"__getstate__", (PyCFunction) SumTree_getstate, METH_VARARGS,
     ""
    },
    {"__setstate__", (PyCFunction) SumTree_setstate, METH_VARARGS,
     ""
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject SumTreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sumtree.SumTree",
    .tp_doc = "SumTree objects",
    .tp_basicsize = sizeof(SumTreeObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = SumTree_new,
    .tp_init = (initproc) SumTree_init,
    .tp_dealloc = (destructor) SumTree_dealloc,
    .tp_members = SumTree_members,
    .tp_methods = SumTree_methods,
};


static PyObject *
sumtree_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyMethodDef SumtreeMethods[] = {
    {"system",  sumtree_system, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef sumtreemodule = {
    PyModuleDef_HEAD_INIT,
    "_fast_sumtree",   /* name of module */
    "bye", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SumtreeMethods
};

PyMODINIT_FUNC
PyInit__fast_sumtree(void)
{
    PyObject *m;

    m = PyModule_Create(&sumtreemodule);
    if (m == NULL)
        return NULL;
    
    if (PyType_Ready(&SumTreeType) < 0)
        return NULL;
    
    Py_INCREF(&SumTreeType);
    if (PyModule_AddObject(m, "SumTree", (PyObject *) &SumTreeType) < 0) {
        Py_DECREF(&SumTreeType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
