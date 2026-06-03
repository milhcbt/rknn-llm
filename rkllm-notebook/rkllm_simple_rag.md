{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a43e0710",
   "metadata": {},
   "source": [
    "# Simple RKLLM RAG (No DB)\n",
    "\n",
    "Store documents in memory, retrieve text chunks, build a prompt, and ask RKLLM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "02852bd8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RKLLM loaded\n"
     ]
    }
   ],
   "source": [
    "#include <stddef.h>\n",
    "#include \"/home/milh/projects/rknn-llm/rkllm-runtime/Linux/librkllm_api/include/rkllm.h\"\n",
    "#include <dlfcn.h>\n",
    "#include <iostream>\n",
    "#include <string>\n",
    "#include <vector>\n",
    "\n",
    "void* rkllm_lib = dlopen(\n",
    "\"/home/milh/projects/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so\",\n",
    "RTLD_NOW | RTLD_GLOBAL);\n",
    "\n",
    "std::cout << (rkllm_lib ? \"RKLLM loaded\" : dlerror()) << std::endl;"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2b5a44bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "using CreateDefaultParamFn = RKLLMParam (*)();\n",
    "using InitFn = int (*)(LLMHandle*, RKLLMParam*, LLMResultCallback);\n",
    "using RunFn = int (*)(LLMHandle, RKLLMInput*, RKLLMInferParam*, void*);\n",
    "using DestroyFn = int (*)(LLMHandle);\n",
    "\n",
    "auto rkllm_createDefaultParam_fn =\n",
    "reinterpret_cast<CreateDefaultParamFn>(dlsym(rkllm_lib,\"rkllm_createDefaultParam\"));\n",
    "\n",
    "auto rkllm_init_fn =\n",
    "reinterpret_cast<InitFn>(dlsym(rkllm_lib,\"rkllm_init\"));\n",
    "\n",
    "auto rkllm_run_fn =\n",
    "reinterpret_cast<RunFn>(dlsym(rkllm_lib,\"rkllm_run\"));\n",
    "\n",
    "auto rkllm_destroy_fn =\n",
    "reinterpret_cast<DestroyFn>(dlsym(rkllm_lib,\"rkllm_destroy\"));"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "88fa6800",
   "metadata": {},
   "outputs": [],
   "source": [
    "static int callback(RKLLMResult* result, void*, LLMCallState state)\n",
    "{\n",
    "    if(result && result->text)\n",
    "        std::cout << result->text << std::flush;\n",
    "\n",
    "    if(state == RKLLM_RUN_FINISH)\n",
    "        std::cout << \"\\n[FINISHED]\\n\";\n",
    "\n",
    "    return 0;\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "18331e92",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "init=0\n",
      "I rkllm: rkllm-runtime version: 1.2.3, rknpu driver version: 0.9.8, platform: RK3588\n",
      "I rkllm: loading rkllm model from /home/milh/models/Qwen3-4B-w8a8-npu.rkllm\n",
      "I rkllm: rkllm-toolkit version: 1.2.1b1, max_context_limit: 4096, npu_core_num: 3, target_platform: RK3588, model_dtype: W8A8\n",
      "I rkllm: Enabled cpus: [4, 5, 6, 7]\n",
      "I rkllm: Enabled cpus num: 4\n"
     ]
    }
   ],
   "source": [
    "LLMHandle handle = nullptr;\n",
    "\n",
    "RKLLMParam param = rkllm_createDefaultParam_fn();\n",
    "param.model_path = \"/home/milh/models/Qwen3-4B-w8a8-npu.rkllm\";\n",
    "\n",
    "int ret = rkllm_init_fn(&handle,&param,callback);\n",
    "std::cout << \"init=\" << ret << std::endl;"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed7c8937",
   "metadata": {},
   "source": [
    "## Knowledge Base"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "25a09b42",
   "metadata": {},
   "outputs": [],
   "source": [
    "std::vector<std::string> docs = {\n",
    "\"RK3588 contains an NPU capable of up to 6 TOPS AI performance.\",\n",
    "\"RKLLM runs quantized language models on Rockchip NPUs.\",\n",
    "\"Qwen3 models can be converted into RKLLM format.\"\n",
    "};"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "87256917",
   "metadata": {},
   "source": [
    "## Retrieval"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c2acfa56",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RK3588 contains an NPU capable of up to 6 TOPS AI performance.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "std::string query =\n",
    "\"How much AI performance does RK3588 provide?\";\n",
    "\n",
    "std::string context;\n",
    "\n",
    "for(const auto& d : docs)\n",
    "{\n",
    "    if(d.find(\"RK3588\") != std::string::npos)\n",
    "        context += d + \"\\n\";\n",
    "}\n",
    "\n",
    "std::cout << context << std::endl;"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b12ec1af",
   "metadata": {},
   "source": [
    "## Build Prompt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ddbd4621",
   "metadata": {},
   "outputs": [],
   "source": [
    "std::string prompt =\n",
    "\"Answer only from the provided context.\\n\\n\"\n",
    "\"Context:\\n\" + context +\n",
    "\"\\nQuestion:\\n\" + query +\n",
    "\"\\nAnswer:\";"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fbfa266b",
   "metadata": {},
   "source": [
    "## Ask Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1e223e13",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Answer: RK3588 provides up to 6 TOPS AI performance.\n",
      "[FINISHED]\n",
      "\n",
      "run=0\n"
     ]
    }
   ],
   "source": [
    "RKLLMInput input{};\n",
    "input.input_type = RKLLM_INPUT_PROMPT;\n",
    "input.prompt_input = prompt.data();\n",
    "\n",
    "RKLLMInferParam infer{};\n",
    "infer.mode = RKLLM_INFER_GENERATE;\n",
    "infer.keep_history = 0;\n",
    "\n",
    "int run_ret =\n",
    "rkllm_run_fn(handle,&input,&infer,nullptr);\n",
    "\n",
    "std::cout << \"\\nrun=\" << run_ret << std::endl;"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7ccff235",
   "metadata": {},
   "outputs": [],
   "source": [
    "rkllm_destroy_fn(handle);"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "C++23",
   "language": "cpp",
   "name": "xcpp23"
  },
  "language_info": {
   "codemirror_mode": "text/x-c++src",
   "file_extension": ".cpp",
   "mimetype": "text/x-c++src",
   "name": "C++",
   "nbconvert_exporter": "",
   "pygments_lexer": "",
   "version": "cxx23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
