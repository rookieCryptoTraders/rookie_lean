import subprocess

def run_data_generation():
    """
    Generates sample market data using the Lean CLI for various security types and resolutions.
    """
    security_types = ["Equity", "Forex", "Cfd", "Future", "Crypto", "Option"]
    resolutions = ["Tick", "Second", "Minute", "Hour", "Daily"]
    quote_trade_ratios = [1,2.0]

    # 2 days of data: 2026-02-05 to 2026-02-06
    start_date = "20260205"
    end_date = "20260207"
    
    symbol_count = 3
    random_seed = 1

    for st in security_types:
        for res in resolutions:
            for quote_trade_ratio in quote_trade_ratios:
                # Construct the lean data generate command
                cmd = [
                    "lean", "data", "generate",
                    "--start", start_date,
                    "--end", end_date,
                    "--symbol-count", str(symbol_count),
                    "--security-type", st,
                    "--resolution", res,
                    "--random-seed", str(random_seed)
                ]
                
                # The --quote-trade-ratio option is only applicable for Option, Future, and Crypto
                if st in ["Option", "Future", "Crypto"]:
                    cmd.extend(["--quote-trade-ratio", str(quote_trade_ratio)])
                
                print(f"\n>>> Generating data for SecurityType: {st}, Resolution: {res}")
                print(f"Command: ")
                print(' '.join(cmd))
                
                try:
                    # Execute the command. Use check=True to raise an exception if the command fails.
                    # Standard output is printed to the console in real-time.
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred while generating data for {st} {res}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_data_generation()
