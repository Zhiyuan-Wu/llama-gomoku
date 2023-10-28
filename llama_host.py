import grpc
from concurrent import futures
import llama_host_pb2
import llama_host_pb2_grpc
import argparse
from llama import Llama
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llama_dir",
        type=str,
        default="llama-2-7b",
        help="The directory where the base model is stored.",
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default="model/checkpoint_n15b6c128i3000d230912.pth",
        # default=None,
        help="The path to the policy_net.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        # default="sft_lora_82K.pth",
        default=None,
        help="The path to the lora patch.",
    )
    parser.add_argument(
        "--projection_path",
        type=str,
        default="model/projection_step100000.pth",
        # default=None,
        help="The path to the projection patch.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="llama-2-7b/tokenizer.model",
        help="The path to the tokenizer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="The top_p."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="The max_seq_len."
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=200,
        help="The max_gen_len."
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=4,
        help="The max_batch_size."
    )
    parser.add_argument(
        "--address",
        type=str,
        default="[::]:17425",
        help="the address that host llama grpc services."
    )
    return parser.parse_args()


class llamahostServicer(llama_host_pb2_grpc.llamahostServicer):
    """Missing associated documentation comment in .proto file."""
    def __init__(self,
                 llama_dir: str = "llama-2-7b",
                 policy_path = None,
                 projection_path = None,
                 lora_path = None,
                 tokenizer_path: str = "tokenizer.model",
                 temperature: float = 0.0,
                 top_p: float = 0.9,
                 max_seq_len: int = 2048,
                 max_gen_len: int = 300,
                 max_batch_size: int = 1,):
        self.generator = Llama.build(
                                    ckpt_dir=llama_dir,
                                    policy_path=policy_path,
                                    projection_path=projection_path,
                                    lora_path=lora_path,
                                    tokenizer_path=tokenizer_path,
                                    max_seq_len=max_seq_len,
                                    max_batch_size=max_batch_size,
                                    )
        self.generator.model.train(False)
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len

    def complete(self, request, context):
        prompts = []
        boards = []
        for x in request.data:
            _data = json.loads(x)
            boards.append(_data["canonicalBoard"])
            prompts.append(_data["prompt"])
        results = self.generator.text_completion(
                                                prompts,
                                                boards,
                                                max_gen_len=self.max_gen_len,
                                                temperature=self.temperature,
                                                top_p=self.top_p,
                                                )
        _output = llama_host_pb2.Output()
        _output.data.extend([x["generation"] for x in results])

        return _output
    
def server_run(args):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    Servicer = llamahostServicer(
        llama_dir=args.llama_dir,
        lora_path=args.lora_path,
        policy_path=args.policy_path,
        projection_path=args.projection_path,
        tokenizer_path=args.tokenizer_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        max_batch_size=args.max_batch_size,
    )
    llama_host_pb2_grpc.add_llamahostServicer_to_server(Servicer, server)
    server.add_insecure_port(args.address)
    server.start()
    print(f"llamahostServicer runing on {args.address}")
    server.wait_for_termination()

if __name__=="__main__":
    args = parse_args()
    server_run(args)