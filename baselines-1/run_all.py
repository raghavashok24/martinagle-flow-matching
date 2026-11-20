"""Run every initial baseline experiment in order.  Usage:  python run_all.py"""
import exp0_toy_sanity, exp1_mot_bounds_lp, exp2_entropic_bridge_ipf
import exp3_defect_discriminator, exp4_heston_synthetic

for mod in (exp0_toy_sanity, exp1_mot_bounds_lp, exp2_entropic_bridge_ipf,
            exp3_defect_discriminator, exp4_heston_synthetic):
    title = mod.__doc__.strip().splitlines()[0]
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
    mod.main()
