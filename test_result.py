from lovart_api import AgentSkill
import json

def test():
    ak = "ak_c37454bc8d320ffdd0191f348f435268"
    sk = "sk_0458f49b696cd118f3a79165374a2df0805ae468ec4abca1d3024fd652ea87de"
    api = AgentSkill(base_url="https://lgw.lovart.ai", access_key=ak, secret_key=sk)
    
    threads = api.get_threads()
    if threads:
        thread_id = threads[0]["id"]
        result = api.get_result(thread_id)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
if __name__ == "__main__":
    test()
