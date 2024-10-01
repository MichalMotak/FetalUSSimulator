from peaceful_pie.unity_comms import UnityComms
import argparse

def run(args) -> None:
    print(args)
    uc = UnityComms(port = args.port)

    uc.Say(message = args.message)
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type = str, required=True)
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    run(args)
