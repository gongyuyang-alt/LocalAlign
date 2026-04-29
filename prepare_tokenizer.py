
import os
import random

import json
import transformers
import torch
import numpy as np

if __name__ == "__main__":
# Delete built-in system prompt (not used in SecAlign training) in Llama-3 series tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct') # Llama 3 series share the same tokenizer
    chat_template = tokenizer.chat_template
    llama_template = """{{- bos_token }}\n
        {%- if custom_tools is defined %}\n{%- set tools = custom_tools %}\n{%- endif %}\n
        {%- if not tools_in_user_message is defined %}\n{%- set tools_in_user_message = true %}\n{%- endif %}\n
        {%- if not date_string is defined %}\n{%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n
        {%- if not tools is defined %}\n{%- set tools = none %}\n{%- endif %}\n\n

        {#- This block extracts the system message, so we can slot it into the right place. #}\n
        {%- if messages[0]['role'] == 'system' %}\n
            {%- set system_message = messages[0]['content']|trim %}\n
            {%- set messages = messages[1:] %}\n
        {%- else %}\n
            {%- set system_message = \"\" %}\n
        {%- endif %}\n\n
        

        {#- System message + builtin tools #}\n
        {%- set system_message_additional = \"\" %}\n
        {%- if builtin_tools is defined or tools is not none %}\n
            {% set system_message_additional = system_message_additional + \"Environment: ipython\\n\" %}\n
        {%- endif %}\n
        {%- if builtin_tools is defined %}\n
            {% set system_message_additional = system_message_additional +  \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"%}\n
        {%- endif %}\n\n

        {%- if tools is not none and not tools_in_user_message %}\n
            {% set system_message_additional = system_message_additional + \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" %}\n
            {% set system_message_additional = system_message_additional + 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' %}\n
            {% set system_message_additional = system_message_additional +  \"Do not use variables.\\n\\n\" %}\n
            {%- for t in tools %}\n
                {% set system_message_additional = system_message_additional + t | tojson(indent=4) %}\n
                {% set system_message_additional = system_message_additional + \"\\n\\n\" %}\n
            {%- endfor %}\n
        {%- endif %}\n\n

        {#- If we have a system message, we put it in the beginning #}\n
        {%- if system_message != \"\" or system_message_additional != \"\" %}\n
            {{- '<|start_header_id|>system<|end_header_id|>\\n\\n' -}}\n
            {{- system_message_additional -}}\n
            {{- system_message + '<|eot_id|>' -}}\n
        {%- endif %}\n\n

        {#- Custom tools are passed in a user message with some extra guidance #}\n
        {%- if tools_in_user_message and not tools is none %}\n
            {#- Extract the first user message so we can plug it in here #}\n
            {%- if messages | length != 0 %}\n
                {%- set first_user_message = messages[0]['content']|trim %}\n
                {%- set messages = messages[1:] %}\n
            {%- else %}\n
                {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n
            {%- endif %}\n
            {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n
            {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n
            {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n
            {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n
            {{- \"Do not use variables.\\n\\n\" }}\n
            {%- for t in tools %}\n
                {{- t | tojson(indent=4) }}\n
                {{- \"\\n\\n\" }}\n
            {%- endfor %}\n
            {{- first_user_message + \"<|eot_id|>\"}}\n
        {%- endif %}\n\n
        
        {%- for message in messages %}\n
            {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n
                {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n
            {%- elif 'tool_calls' in message %}\n
                {%- if not message.tool_calls|length == 1 %}\n
                    {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n
                {%- endif %}\n
                {%- set tool_call = message.tool_calls[0].function %}\n
                {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n
                    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n
                    {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n
                    {%- for arg_name, arg_val in tool_call.arguments | items %}\n
                        {{- arg_name + '=\"' + arg_val + '\"' }}\n
                        {%- if not loop.last %}\n{{- \", \" }}\n{%- endif %}\n
                    {%- endfor %}\n
                    {{- \")\" }}\n
                {%- else  %}\n
                    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n
                    {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n
                    {{- '\"parameters\": ' }}\n
                    {{- tool_call.arguments | tojson }}\n
                    {{- \"}\" }}\n
                {%- endif %}\n
                {%- if builtin_tools is defined %}\n
                    {#- This means we're in ipython mode #}\n
                    {{- \"<|eom_id|>\" }}\n
                {%- else %}\n
                    {{- \"<|eot_id|>\" }}\n
                {%- endif %}\n
            {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n
                {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n
                {%- if message.content is mapping or message.content is iterable %}\n
                    {{- message.content | tojson }}\n
                {%- else %}\n
                    {{- message.content }}\n
                {%- endif %}\n
                {{- \"<|eot_id|>\" }}\n
            {%- endif %}\n
        {%- endfor %}\n
        
        {%- if add_generation_prompt %}\n{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n
    """

    Qwen_template = """{{- bos_token }}

{%- for message in messages %}
    {%- if message.role == "user" %}
<|im_start|>user
{{ message.content }}
<|im_end|>

    {%- elif message.role == "input" %}
<|im_start|>input
{{ message.content }}
<|im_end|>

    {%- elif message.role == "assistant" %}
<|im_start|>assistant
{{ message.content }}
<|im_end|>

    {%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}"""

    tokenizer.chat_template=Qwen_template
    tokenizer.save_pretrained('./github_repo/Meta_SecAlign/data/qwen_3_tokenizer')
    with open("./github_repo/Meta_SecAlign/data/qwen_3_tokenizer/chat_template.jinja", 'w') as file:
        file.write(tokenizer.chat_template)