import traceback
from main_code import main
from simulator.util_functions import get_args
from time import time
import warnings

def deepevolve_interface():
    args = get_args()
    
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start_time = time()
            indi, results = main(args)
            runtime = time() - start_time
        warning_messages = [str(w.message) for w in caught]
        runtime = round(runtime / 60, 2)
        if indi == -1:
            error_info = f"""
                Error type: Train Loss is NaN
                Error message: Training loss has becom NaN. Possible causes include exploding gradients, invalid model outputs, or numerical instability.
                Traceback: No traceback (loss value checked before backward)."""
            return False, error_info
        
        metrics = {
            "runtime_minutes": runtime,
            **results
        }
        if warning_messages:
            warning_messages = list(set(warning_messages))
            if len(warning_messages) > 10:
                warning_messages = warning_messages[:10]
            metrics["program_warnings"] = warning_messages
        return True, metrics
    except Exception as e:
        error_traceback = traceback.format_exc()
        error_info = f"""
            Error type: {type(e).__name__}
            Error message: {str(e)}
            Traceback: {error_traceback}
        """
        return False, error_info

if __name__ == "__main__":
    status, results = deepevolve_interface()
    print(f"Status: {status}")
    print(f"Results: {results}")