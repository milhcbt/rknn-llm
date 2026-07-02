/*
g++ rkllm_demo.cpp \
-I/usr/local/include \
-L/usr/lib \
-lrkllmrt \
-o rkllm_demo
---
./rkllm_demo deepseek.rkllm 1024 4096

*/

#include <string.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <csignal>
#include <vector>

#include "rkllm.h"

using namespace std;

LLMHandle llmHandle = nullptr;

void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        cout << "\nExiting..." << endl;

        LLMHandle tmp = llmHandle;
        llmHandle = nullptr;

        rkllm_destroy(tmp);
    }

    exit(signal);
}

int callback(
    RKLLMResult *result,
    void *userdata,
    LLMCallState state)
{
    if (state == RKLLM_RUN_NORMAL)
    {
        if (result && result->text)
        {
            printf("%s", result->text);
            fflush(stdout);
        }
    }
    else if (state == RKLLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        printf("\n[Generation error]\n");
    }

    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cout
            << "Usage:\n"
            << argv[0]
            << " model.rkllm max_new_tokens max_context_len"
            << endl;

        return 1;
    }

    signal(SIGINT, exit_handler);

    cout << "RKLLM init..." << endl;

    RKLLMParam param = rkllm_createDefaultParam();

    param.model_path = argv[1];

    // Sampling
    param.top_k = 20;
    param.top_p = 0.8;
    param.temperature = 0.5;
    param.repeat_penalty = 1.25;
    param.frequency_penalty = 0.0;
    param.presence_penalty = 0.0;

    param.max_new_tokens =
        atoi(argv[2]);

    param.max_context_len =
        atoi(argv[3]);

    // Hide internal tokens
    param.skip_special_token = true;

    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash = 1;

    RKLLMCallback cb = {};
    cb.result_callback = callback;

    int ret =
        rkllm_init(
            &llmHandle,
            &param,
            &cb
        );

    if (ret != 0)
    {
        cout << "RKLLM init failed" << endl;
        return -1;
    }

    cout << "RKLLM init success" << endl;

    vector<string> examples;

    examples.push_back(
        "What is the capital of France?"
    );

    examples.push_back(
        "Explain neural networks simply."
    );

    cout
        << "\n========== RKLLM Chat ==========\n"
        << endl;

    for (int i = 0; i < examples.size(); i++)
    {
        cout
            << "[" << i << "] "
            << examples[i]
            << endl;
    }

    cout
        << "\nCommands:"
        << "\n  clear  - clear conversation history"
        << "\n  exit   - quit"
        << "\n================================\n"
        << endl;

    RKLLMInput input;
    memset(
        &input,
        0,
        sizeof(RKLLMInput)
    );

    RKLLMInferParam infer;
    memset(
        &infer,
        0,
        sizeof(RKLLMInferParam)
    );

    infer.mode = RKLLM_INFER_GENERATE;

    // Keep conversation context
    infer.keep_history = 1;

    while (true)
    {
        string text;

        cout << "\nYou> ";

        getline(cin, text);

        if (text == "exit")
        {
            break;
        }

        if (text == "clear")
        {
            rkllm_clear_kv_cache(
                llmHandle,
                1,
                nullptr,
                nullptr
            );

            cout
                << "Conversation cleared"
                << endl;

            continue;
        }

        for (int i = 0; i < examples.size(); i++)
        {
            if (text == to_string(i))
            {
                text = examples[i];
            }
        }

        input.input_type =
            RKLLM_INPUT_PROMPT;

        input.role =
            "user";

        input.prompt_input =
            text.c_str();

        cout << "AI> ";

        rkllm_run(
            llmHandle,
            &input,
            &infer,
            nullptr
        );
    }

    rkllm_destroy(llmHandle);

    return 0;
}
