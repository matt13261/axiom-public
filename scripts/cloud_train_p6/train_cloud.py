#!/usr/bin/env python3
"""
Training MCCFR cloud pour AbstractionCartesV2 (blueprint P6).

Usage sur la VM :
    python train_cloud.py --iterations 5000000 --output blueprint_v2.pkl
    python train_cloud.py --iterations 5000000 --output blueprint_v2.pkl --checkpoint-every 500000
"""
import argparse
import sys
import time
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _worker_noop():
    pass


def _worker_mccfr(worker_id, nb_iterations, stacks, queue, batch_id):
    """Worker process : crée son MCCFRHoldEm, entraîne, renvoie noeuds via queue."""
    try:
        dossier = str(Path(__file__).parent.parent.parent)
        if dossier not in sys.path:
            sys.path.insert(0, dossier)
        from ai.mccfr import MCCFRHoldEm
        mccfr = MCCFRHoldEm()
        if nb_iterations > 0:
            mccfr.entrainer(
                nb_iterations=nb_iterations,
                stacks=stacks,
                verbose=False,
                save_every=0,
            )
        queue.put((worker_id, batch_id, mccfr.noeuds, None))
    except Exception as e:
        import traceback
        queue.put((worker_id, batch_id, None,
                   f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _fusionner_noeuds(dicts_list, base_dict=None):
    from ai.mccfr import NoeudCFR
    fusionne = base_dict if base_dict is not None else {}
    for noeuds in dicts_list:
        for cle, noeud in noeuds.items():
            if cle not in fusionne:
                nouveau = NoeudCFR(noeud.nb_actions)
                nouveau.regrets_cumules = list(noeud.regrets_cumules)
                nouveau.strategie_somme = list(noeud.strategie_somme)
                nouveau.nb_visites      = noeud.nb_visites
                fusionne[cle] = nouveau
            else:
                n = fusionne[cle]
                if noeud.nb_actions > n.nb_actions:
                    diff = noeud.nb_actions - n.nb_actions
                    n.regrets_cumules.extend([0.0] * diff)
                    n.strategie_somme.extend([0.0] * diff)
                    n.nb_actions = noeud.nb_actions
                for i in range(min(noeud.nb_actions, n.nb_actions)):
                    n.regrets_cumules[i] += noeud.regrets_cumules[i]
                    n.strategie_somme[i] += noeud.strategie_somme[i]
                n.nb_visites += noeud.nb_visites
    return fusionne


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--iterations',       type=int, required=True)
    parser.add_argument('--output',           type=str, required=True)
    parser.add_argument('--checkpoint-every', type=int, default=500_000)
    parser.add_argument('--stacks',           type=int, default=1500)
    parser.add_argument('--workers',          type=int, default=1)
    parser.add_argument('--batch-size',       type=int, default=1_000_000)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from ai.mccfr import MCCFRHoldEm
    from ai.strategy import sauvegarder_blueprint

    print(f"=== Cloud Training MCCFR — AbstractionCartesV2 ===")
    print(f"Iterations : {args.iterations:,} | Output : {args.output}")
    print(f"Start : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    trainer = MCCFRHoldEm()
    if trainer._abs_cartes.centroides is None:
        print("ERREUR : centroïdes V2 absents — copier centroides_v2.npz dans data/abstraction/")
        sys.exit(1)

    start = time.time()

    batch_size = min(args.batch_size, args.iterations)
    n_batches = (args.iterations + batch_size - 1) // batch_size
    checkpoint_dir = output_path.parent
    blueprint_global = None

    workers = max(1, args.workers)
    iter_per_worker = batch_size // workers

    for batch_id in range(1, n_batches + 1):
        batch_start = time.time()
        print(f"[Batch {batch_id}/{n_batches}] Spawn {workers} workers × {iter_per_worker:,} iter...")
        sys.stdout.flush()

        queue = mp.Queue()
        procs = []
        for w_id in range(workers):
            p = mp.Process(
                target=_worker_mccfr,
                args=(w_id, iter_per_worker, args.stacks, queue, batch_id),
            )
            p.start()
            procs.append(p)

        results = []
        errors = []
        recus = 0
        while recus < workers:
            try:
                w_id, b_id, noeuds, err = queue.get(timeout=1.0)
                recus += 1
                if err:
                    errors.append((w_id, err))
                elif noeuds is not None:
                    results.append(noeuds)
            except Exception:
                if not any(p.is_alive() for p in procs):
                    break

        for p in procs:
            p.join()

        if errors:
            print(f"[Batch {batch_id}] ERREURS workers : {errors}")
            sys.exit(1)

        blueprint_global = _fusionner_noeuds(results, base_dict=blueprint_global)

        checkpoint_path = checkpoint_dir / f"{output_path.stem}_batch_{batch_id}.pkl"
        sauvegarder_blueprint(blueprint_global, str(checkpoint_path))
        batch_duration = time.time() - batch_start
        it_par_sec = batch_size / max(batch_duration, 1e-9)
        print(f"[Batch {batch_id}/{n_batches}] Terminé en {batch_duration/60:.2f} min "
              f"({it_par_sec:.0f} it/s) | {len(blueprint_global):,} infosets — {checkpoint_path.name}")
        sys.stdout.flush()

    sauvegarder_blueprint(blueprint_global, str(output_path))

    duration = time.time() - start
    print(f"\n=== Terminé en {duration/3600:.2f}h | {len(blueprint_global):,} infosets | {args.iterations/max(duration,1e-9):.0f} it/s ===")


if __name__ == '__main__':
    main()
