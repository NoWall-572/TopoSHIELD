import argparse
import sys
import asyncio
from tqdm import tqdm
from sandbox.simulator import Simulator

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

async def run(args):
    sim = Simulator(
        instance_id=args.instance_id, 
        attack_method=args.attack_method, 
        model=args.model, 
        topo=args.topo, 
        defense=args.defense, 
        defense_method=args.defense_method
    )
    agent_list = sim.initialize()
    await sim.emulate(
        args.time_step, 
        agent_list, 
        print_prompt=args.print_prompt, 
        print_log=args.print_log
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent System Simulator Runner")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=108)
    parser.add_argument("--instance_id", type=int, default=0)
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--time_step", type=int, default=3)
    parser.add_argument("--attack_method", default="vanilla", choices=['vanilla', 'inject', 'tool', 'rag'])
    parser.add_argument("--defense", action="store_true") 
    
    parser.add_argument("--defense_method", default="none", 
                        choices=['none', 'toposhield'], 
                        help="Choose defense strategy")
    
    parser.add_argument("--topo", default="auto")
    parser.add_argument("--print_prompt", action="store_true")
    parser.add_argument("--print_log", action="store_true")

    args = parser.parse_args()

    if args.defense and args.defense_method == "none":
        args.defense_method = "toposhield"

    ids_to_run = range(args.start_index, args.end_index)
    if args.instance_id != 0:
        ids_to_run = [args.instance_id]

    print(f"[*] Starting: Model={args.model}, Attack={args.attack_method}, Defense={args.defense_method}")

    for i in tqdm(ids_to_run, desc="Total Progress"):
        try:
            args.instance_id = i
            asyncio.run(run(args))
        except Exception as e:
            print(f"[!] Error in instance {i}: {e}")
            continue