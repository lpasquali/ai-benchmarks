import json
import subprocess
import time
from openai import OpenAI

# Point to your Vast.ai Ollama instance
client = OpenAI(
    base_url="http://<YOUR_VAST_IP>:11434/v1",
    api_key="ollama" 
)
MODEL = "llama3.1:405b" 

# --- 1. Define the Tools ---
def execute_shell(command):
    print(f"\n[AGENT EXECUTING]: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr}"

def create_kind_cluster():
    config = "kind: Cluster\napiVersion: kind.x-k8s.io/v1alpha4\nnetworking:\n  disableDefaultCNI: true"
    with open("kind-config.yaml", "w") as f:
        f.write(config)
    return execute_shell("kind create cluster --config kind-config.yaml")

def install_cilium():
    execute_shell("helm repo add cilium [https://helm.cilium.io/](https://helm.cilium.io/)")
    execute_shell("helm repo update")
    return execute_shell("helm install cilium cilium/cilium --version 1.15.0 --namespace kube-system")

def verify_k8s_state():
    time.sleep(10) 
    return execute_shell("kubectl wait --for=condition=Ready nodes --all --timeout=300s")

def install_capi_proxmox():
    return execute_shell("clusterctl init --infrastructure proxmox")

def verify_capi_state():
    time.sleep(10)
    return execute_shell("kubectl wait --for=condition=Available deployment -l cluster.x-k8s.io/provider=infrastructure-proxmox -A --timeout=300s")

def provision_proxmox_cluster():
    execute_shell("kubectl apply -f proxmox-secret.yaml")
    return execute_shell("clusterctl generate cluster proxmox-workload | kubectl apply -f -")

available_tools = {
    "create_kind_cluster": create_kind_cluster,
    "install_cilium": install_cilium,
    "verify_k8s_state": verify_k8s_state,
    "install_capi_proxmox": install_capi_proxmox,
    "verify_capi_state": verify_capi_state,
    "provision_proxmox_cluster": provision_proxmox_cluster
}

# --- 2. Define the LLM Schema ---
tools_schema = [
    {"type": "function", "function": {"name": "create_kind_cluster", "description": "Creates a Kind cluster with CNI disabled."}},
    {"type": "function", "function": {"name": "install_cilium", "description": "Installs the Cilium CNI via Helm."}},
    {"type": "function", "function": {"name": "verify_k8s_state", "description": "Waits for all K8s nodes to be Ready. Must be run after installing Cilium."}},
    {"type": "function", "function": {"name": "install_capi_proxmox", "description": "Initializes Cluster API with the Proxmox provider."}},
    {"type": "function", "function": {"name": "verify_capi_state", "description": "Verifies CAPI controllers are running."}},
    {"type": "function", "function": {"name": "provision_proxmox_cluster", "description": "Applies secrets and generates the workload cluster."}}
]

system_prompt = """You are an SRE agent. Your job is to provision a Proxmox cluster using CAPI on Kind.
You must use your tools in this exact order:
1. create_kind_cluster
2. install_cilium
3. verify_k8s_state
4. install_capi_proxmox
5. verify_capi_state
6. provision_proxmox_cluster
Do not skip steps. If a tool returns an ERROR, stop and report it."""

# --- 3. The Execution Loop ---
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Begin the Proxmox CAPI deployment pipeline."}
]

print("Starting SRE Agent Pipeline...")

while True:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    messages.append(response_message)
    
    if not response_message.tool_calls:
        print(f"\n[AGENT FINAL RESPONSE]: {response_message.content}")
        break

    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        print(f"\n[LLM REQUESTS TOOL]: {function_name}")
        
        function_to_call = available_tools[function_name]
        function_response = function_to_call()
        
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": str(function_response),
        })