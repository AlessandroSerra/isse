# Mappa del progetto ISSE

> Scopo: fornire a un agente successivo una vista rapida, mirata ed estendibile della repo, evitando una nuova analisi completa prima di riprendere il lavoro.

## 1. Identità del progetto

- **Nome package**: `isse`
- **Descrizione**: *Ichnusa Solid State Environment*, ambiente Python per workflow di fisica dello stato solido e simulazioni atomistiche.
- **Layout**: package in `src/isse`.
- **Python**: `>=3.11`
- **Build backend**: `uv_build`
- **Dipendenze runtime**: `numpy`, `spglib`, `numba`
- **Unità interne usate dai parser/workflow**:
  - lunghezze/posizioni/celle: Å
  - velocità: Å/fs
  - masse: amu
  - forze: eV/Å

## 2. Stato rapido della repo

- `README.md` è minimale.
- `src/isse/__init__.py`, `src/isse/helpers/__init__.py`, `src/isse/io/__init__.py` sono vuoti: non espongono ancora API pubbliche aggregate.
- `src/isse/convert_file.py` è ora una CLI/converter basata su **ASE** con test di equivalenza integrati; `ase` e `scipy` non sono però nelle dipendenze runtime di `pyproject.toml`.
- `src/isse/convert.py` è un converter nuovo/non tracciato che reimplementa la conversione senza ASE, usando `Atoms`, `Trajectory` e i moduli I/O e writer ISSE per i formati supportati internamente.
- `src/isse/radial_distribution.py` e `src/isse/helpers/periodic.py` aggiungono calcolo RDF e utility PBC/minimum-image.
- Sono presenti file in `tests/`, ma `.gitignore` ignora `tests/`: quindi non sono versionati. `.gitignore` ignora anche `dist/`, `archive/` e `src/isse/TODO.md`.
- Nota working tree dopo questo aggiornamento: `AGENTS.md`, `src/isse/phonon_temperatures.py` e `src/isse/project_velocities.py` modificati; `src/isse/parsers/` risulta rimosso e `src/isse/io/` aggiunto; `src/isse/convert.py` non tracciato (`git status --short`).

## 3. Albero logico

```text
src/isse/
├── constants.py              # costanti fisiche, conversioni unità, masse atomiche
├── structures.py             # Atoms e Trajectory lazy
├── phonon_temperatures.py    # entry point alto livello per temperature modali
├── project_velocities.py     # proiezione velocità su modi fononici
├── convert_file.py           # CLI/converter ASE con test di equivalenza
├── convert.py                # converter ISSE-native senza ASE, non tracciato
├── radial_distribution.py    # RDF totale su Trajectory lazy
├── helpers/
│   ├── cell_mapping.py       # mapping atomi supercella -> cella primitiva + basis
│   ├── periodic.py           # wrap/unwrap e minimum-image in PBC
│   └── symmetry.py           # scaled positions, primitive cell via spglib, qpoint helpers
└── io/
    ├── parse_alamode.py      # lettura file ALAMODE .evec
    ├── parse_gpumddump.py    # parser lazy GPUMD dump
    ├── parse_lammps.py       # parser LAMMPS data/dump
    ├── parse_vasp.py         # parser POSCAR
    ├── write_gpumddump.py    # writer GPUMD/extended XYZ
    ├── write_lammps.py       # writer LAMMPS data/dump
    └── write_vasp.py         # writer POSCAR
```

## 4. Modello dati centrale

### `src/isse/structures.py`

#### `Atoms`
Dataclass `slots=True` che rappresenta una configurazione atomistica.

Campi principali:
- `symbols: list[str]`
- `cell: np.ndarray`, shape `(3, 3)`, vettori di cella per riga
- `positions: np.ndarray`, shape `(n_atoms, 3)`
- opzionali: `unwrapped_positions`, `velocities`, `masses`, `forces`
- `arrays: dict[str, np.ndarray]` per array per-atomo extra, es. `id`, `type`, `groups`
- `info: dict[str, str]` per metadata, es. `timestep`, header POSCAR

Validazione in `__post_init__` su shape. Proprietà booleane: `has_unwrapped_positions`, `has_velocities`, `has_masses`, `has_forces`.

#### `Trajectory`
Sequenza lazy di `Atoms` basata su offset byte in un file e una funzione `reader(path, offset)`.

Caratteristiche:
- `__getitem__(int)` legge un frame singolo.
- `__getitem__(slice)` restituisce una nuova `Trajectory` lazy sugli offset selezionati.
- `__iter__()` itera leggendo frame on demand.
- Memoria proporzionale al numero di frame, non alla dimensione del file.

## 5. Flussi principali

### 5.1 I/O -> dati atomistici

```text
file esterno
  ├─ POSCAR              -> parse_poscar(...)       -> Atoms
  ├─ LAMMPS data         -> parse_lammps_data(...)  -> Atoms
  ├─ LAMMPS dump         -> parse_lammps_dump(...)  -> Trajectory lazy
  └─ GPUMD dump          -> parse_gpumd_dump(...)   -> Trajectory lazy
```

### 5.2 Proiezione velocità su fononi

```text
Trajectory + reference Atoms + ALAMODE .evec
    │
    ├─ read_alamode_evec(evec_filepath)
    │     -> primitive_cell, qpoints, eigenvalues, eigenvectors
    │
    ├─ map_atoms_to_primitive(reference_atoms)
    │     -> cell_indices, basis_indices
    │
    ├─ _precompute_coefficients(...)
    │     -> coefficienti complessi frame-independent
    │
    ├─ _iter_velocity_batches(trajectory, natoms, batch_size)
    │     -> batch velocità, shape (nframes_batch, natoms, 3)
    │
    ├─ _project_batch_numba oppure _project_batch_numpy
    │     -> qdot2, atomic_norms
    │
    └─ _compute_parseval_errors(...)
          -> errori Parseval per frame
```

Funzione pubblica: `project_velocities(...)` in `src/isse/project_velocities.py`.

### 5.3 Temperature modali

```text
calculate_temperature(...)
    └─ project_velocities(...)
         └─ qpoints, qdot2, atomic_norms, parseval_errors
    └─ converte <qdot2> medio in temperature usando
       AMU_A2_FS2_TO_EV / KB_EV_K
```

Output dict attuale:
- `qpoints`, shape `(nqpoints, 3)`
- `mode_temperatures`, shape `(nqpoints, nmodes)`
- `mean_mode_temperature`
- opzionale `selected_mode_temperatures`, shape `(nframes, nselected, nmodes)`

Nota: `reconstructed_temperature` viene calcolata e loggata, ma non inserita in `results`; la docstring è stata aggiornata per riflettere l'output effettivo.

## 6. Moduli e responsabilità

### `constants.py`

Contiene:
- conversioni: `ANGSTROM_TO_BOHR`, `BOHR_TO_ANGSTROM`, `AMU_A2_FS2_TO_EV`, `KB_EV_K`, `KCAL_MOL_TO_EV`, `PS_TO_FS`
- tabella `ATOMIC_MASSES`
- helper:
  - `mass_from_symbol(symbol)`
  - `masses_from_symbols(symbols)`
  - `symbol_from_mass(mass, tolerance=1e-3)`
  - `symbols_from_masses(masses, tolerance=1e-3)`

### `io/parse_lammps.py`

API:
- `parse_lammps(filename, format="dump"|"data", symbols=None, units=None)` dispatcher
- `parse_lammps_data(...) -> Atoms`
- `parse_lammps_dump(...) -> Trajectory`

Note:
- Supporta `units="metal"` e `units="real"`.
- Data file: supporto dichiarato per `atom_style atomic`, sezioni `Masses` e `Velocities` opzionali.
- Dump: supporta posizioni `x/y/z`, scaled `xs/ys/zs`, unwrapped `xu/yu/zu`, velocità, forze, masse, `id`, `type`.
- Box ortogonali e triclinici convertiti in cella 3x3 con vettori per riga.

### `io/parse_gpumddump.py`

API:
- `parse_gpumd_dump(filename) -> Trajectory`
- reader interno `_read_frame_gpumd_dump(filepath, offset) -> Atoms`

Note:
- Richiede header con `Lattice` e `Properties`.
- Legge proprietà tipo extended XYZ: `species`, `pos`, opzionali `vel`, `force`, `unwrapped_position`, `mass`, `group`, `Time`.

### `io/parse_vasp.py`

API:
- `parse_poscar(filename) -> Atoms`

Note:
- Legge POSCAR con scaling factor positivo.
- Supporta coordinate Direct e Cartesian/Kartesian.
- Calcola masse dai simboli tramite `mass_from_symbol`.
- Ignora/gestisce parzialmente Selective dynamics.

### `io/parse_alamode.py`

API:
- `read_alamode_evec(filename) -> (primitive_cell, qpoints, eigenvalues, eigenvectors)`

Note:
- Legge file ALAMODE `.evec`.
- Converte i vettori di reticolo da Bohr ad Å.
- `eigenvectors` shape `(nq, nmodes, nat_primitive, 3)` complessa.

### `io/write_lammps.py`

API:
- `write_lammps_data(filename, atoms, units="metal") -> None`
- `write_lammps_dump(filename, trajectory, units="metal", fractional=False) -> None` con `trajectory` = `Atoms | Trajectory | list[Atoms]`

Note:
- Scrive LAMMPS data `atom_style atomic` single-frame.
- Scrive LAMMPS dump multi-frame da iterabile di `Atoms`.
- Converte velocità/forze da unità interne verso `metal` o `real`.
- Richiede celle in forma restricted-triclinic.

### `io/write_gpumddump.py`

API:
- `write_gpumd_dump(filename, trajectory) -> None` con `trajectory` = `Atoms | Trajectory | list[Atoms]`

Note:
- Scrive formato GPUMD/extended XYZ multi-frame.
- Include proprietà opzionali se presenti: `vel`, `force`, `unwrapped_position`, `mass`, `group`, `Time`.

### `io/write_vasp.py`

API:
- `write_poscar(filename, atoms, direct=True) -> None`

Note:
- Scrive POSCAR single-frame.
- Raggruppa le righe di posizione per specie in accordo con le righe specie/conteggi.
- Supporta coordinate Direct o Cartesian.

### `helpers/symmetry.py`

API/helper:
- `get_scaled_positions(atoms)` oppure `get_scaled_positions(positions, cell)`
- `find_primitive_cell(atoms, tolerance=1e-5) -> Atoms` usando `spglib`
- `_get_supercell_transofm_matrix(supercell, primitive_cell, tolerance=1e-6)` *(privata; typo nel nome: `transofm`)*
- `_generate_qpoints(supercell)` *(privata)*

Note:
- `find_primitive_cell` solleva errore se non trova una cella più piccola; quindi una struttura già primitiva viene trattata come caso di errore.

### `helpers/cell_mapping.py`

API:
- `map_atoms_to_primitive(atoms, primitive_cell=None, basis=None, tolerance=1e-3)`

Responsabilità:
- Mappa ogni posizione cartesiana come:
  `position = (cell_index + basis[basis_index]) @ primitive_cell`
- Se `primitive_cell` e `basis` non sono forniti, usa `find_primitive_cell`.
- Restituisce `(cell_indices, basis_indices), residuals`.
- Verifica residui e popolazioni per sito di basis.

### `project_velocities.py`

API principale:
- `project_velocities(trajectory, reference_atoms, evec_filepath, batch_size=100, parseval_tolerance=1e-6)`

Helper:
- `_compute_parseval_errors(qdot2, atomic_norms)`
- `_precompute_coefficients(qpoints, eigenvectors, cell_indices, basis_indices, masses)`
- `_iter_velocity_batches(trajectory, natoms, batch_size)`
- `_project_batch_numpy(...)`
- `_project_batch_numba(...)` se `numba` disponibile

Contratti importanti:
- `reference_atoms.masses` deve essere presente.
- Ogni frame della trajectory deve contenere `velocities` shape `(natoms, 3)`.
- `parseval_tolerance` può essere `None` a runtime, anche se type hint attuale è `float`.

### `phonon_temperatures.py`

API:
- `calculate_temperature(trajectory, reference_atoms, evec_filepath, selected_iqs=None, batch_size=100, parseval_tolerance=1e-6)`

Responsabilità:
- Usa `project_velocities`.
- Calcola temperature modali e temperatura ricostruita.
- Esclude i primi 3 modi a Gamma dalla media termica (`thermal_mask[0, :3] = False`).

Note attuali:
- Calcola `reconstructed_temperature`, ma non la restituisce.
- Restituisce `mean_mode_temperature`; la docstring è allineata all'output effettivo.

### `radial_distribution.py`

API:
- `calculate_rdf(trajectory, r_max, dr, batch_size=100, use_numba=True) -> dict[str, NDArray[np.float64]]`

Responsabilità:
- Calcola la radial distribution function totale `g(r)` su una `Trajectory` lazy.
- Processa posizioni/celle a batch.
- Usa backend Numba se disponibile, altrimenti fallback NumPy.

Contratti/assunzioni:
- Ogni frame deve avere posizioni shape `(natoms, 3)` e cella `(3, 3)`.
- Tutti i frame devono avere lo stesso numero di atomi.
- Output: `r`, `g_r`, `counts`.
- Normalizzazione con volume medio della traiettoria.

Dipendenze interne:
- usa `helpers.periodic.minimum_image_distances` nel fallback NumPy.

### `helpers/periodic.py`

API/helper:
- `minimum_image_displacements(displacements, cell)`
- `minimum_image_distances(displacements, cell)`
- `wrap_positions(positions, cell)`
- `unwrap_positions(positions, cells)`

Responsabilità:
- Utility PBC con celle a vettori per riga (`cartesian = fractional @ cell`).
- Minimum-image, wrap di posizioni cartesiane, unwrap di una traiettoria.

Contratti/assunzioni:
- Celle shape `(3, 3)`; per `unwrap_positions`, celle fisse `(3, 3)` o per-frame `(n_frames, 3, 3)`.
- Minimum-image veloce via coordinate frazionarie arrotondate; può non trovare l'immagine cartesiana più vicina per celle fortemente skewed/non ridotte.

### `convert_file.py`

API/CLI:
- `convert(infile, outfile, infile_type, outfile_type, replicate=None, fractional=False) -> ase.Atoms`
- `main()` entry point script

Responsabilità:
- Converter legacy tramite ASE.
- Supporta input/output custom `alm.lmp`, `lammpstrj`; input custom `alm.xyz`.
- Supporta supercelle (`--replicate`), coordinate frazionarie (`--frac`), stampa POSCAR-style di coordinate scalate (`--print-scaled`).
- Esegue test di equivalenza opzionali: conteggio atomi, composizione, parametri reticolari, Minkowski, spglib, coordinate frazionarie via KDTree/scipy, Kabsch.

Punti aperti:
- Dipende da `ase` e opzionalmente da `scipy`, ma questi pacchetti non sono dichiarati in `pyproject.toml`.
- È script-style dentro il package e non risulta esposto come console script.

### `convert.py`

API:
- `convert(infile, outfile, infile_type, outfile_type, input_units=None, output_units=None, symbols=None, replicate=None, fractional=False, frame=None) -> Atoms | Trajectory`
- `read_file(filename, file_type, units=None, symbols=None) -> Atoms | Trajectory`
- `write_file(filename, atoms_or_trajectory, file_type, units=None, fractional=False) -> None`

Dipendenze interne:
- legge tramite `io/parse_*.py`.
- scrive delegando a `io/write_vasp.py`, `io/write_lammps.py`, `io/write_gpumddump.py`.
- i writer di traiettoria accettano un parametro `trajectory`, supportano anche un singolo `Atoms`, e hanno docstring in stile NumPy.

Responsabilità:
- Conversione ISSE-native senza ASE, usando `Atoms`/`Trajectory` e i moduli I/O interni.
- Formati supportati: `poscar`/`vasp`, `lammps-data`, `lammps-dump`/`lammpstrj`, `gpumd-dump`/`extxyz`.
- Supporta traiettorie lazy, selezione frame, supercelle, coordinate frazionarie dove sensato e unità LAMMPS input/output (`metal`/`real`).
- Include test di equivalenza leggeri e dependency-free: numero frame, numero atomi, composizione, cella, posizioni frazionarie in stesso ordine.

Contratti/assunzioni:
- LAMMPS output usa celle in forma restricted-triclinic compatibile con i parser ISSE.
- Le unità LAMMPS non hanno default: l'utente deve specificare esplicitamente `input_units`/`output_units` o `units` per evitare assunzioni silenziose.
- `lammps-data`, POSCAR/VASP sono output single-frame; per input trajectory multi-frame serve `--frame`.
- Per LAMMPS data, i simboli chimici sono recuperabili in lettura solo passando `symbols`; i test tentano di derivarli dai simboli originali.

## 7. Punti di attenzione / debito tecnico

Questa sezione va aggiornata man mano che il codice evolve.

1. **TODO esistente** (`src/isse/TODO.md`):
   - in `calculate_temperature`, se `reference_atoms` non ha masse, oggi viene sollevata eccezione; desiderato: prenderle dalla trajectory.
   - TODO dice anche che `calculate_temperature` dovrebbe restituire i qpoints, ma il codice attuale li include già in `results["qpoints"]`.
2. **API package non esposta**: gli `__init__.py` sono vuoti; un utente deve importare dai moduli profondi.
3. **Assenza test**: nessuna suite test versionata. Priorità alta per parser, unità, shape, lazy loading, Parseval.
4. **`convert_file.py` / dipendenze CLI**: il converter è stato migrato nel file principale, ma usa `ase` e `scipy` non dichiarati. Decidere se:
   - aggiungerli a dipendenze/extra opzionali,
   - spostare la CLI in un extra/tool separato,
   - esporre un console script ufficiale.
5. **`convert.py` non tracciato**: implementato come converter ISSE-native, ma resta da tracciare/versionare e decidere se esporlo come console script ufficiale.
6. **Typo in `symmetry.py`**: `_get_supercell_transofm_matrix` dovrebbe probabilmente essere `_get_supercell_transform_matrix`.
7. **`parse_vasp.py` Selective dynamics**: la gestione è fragile; dopo una riga `Selective dynamics`, legge il tipo coordinate successivo, ma il ramo `elif coords_type.startswith("s")` resta nel codice e può lasciare `positions` non definito in casi anomali.
8. **`find_primitive_cell`**: se l'input è già primitivo solleva errore. Verificare se questo è desiderato per i workflow di mapping.
9. **Temperature API**: `calculate_temperature` calcola e logga `reconstructed_temperature`, ma non la restituisce; valutare se aggiungerla al dizionario risultati.
10. **Logging**:
   - typo in messaggi tipo `Succesfully`.
   - in `phonon_temperatures.py` il messaggio `logger.info` sembra mancare una parentesi `)` nel testo formattato.
11. **Type hints**:
   - `parseval_tolerance` documentato come opzionale/`None`, ma annotato `float`.
   - alcuni return/import usano `np.ndarray` generico invece di `NDArray[...]`.

## 8. Come estendere questa mappa

Quando aggiungi codice, aggiorna solo le sezioni pertinenti seguendo questo schema:

```markdown
### `path/to/module.py`

API pubblica:
- `funzione_o_classe(...) -> ...`

Responsabilità:
- ...

Contratti/assunzioni:
- shape array, unità, errori attesi, lazy/eager, ecc.

Dipendenze interne:
- usa `...`

Punti aperti:
- ...
```

Per nuovi workflow, aggiungi un diagramma breve in **Flussi principali**.
Per nuovi moduli I/O, specifica sempre:
- formato input/output
- unità lette/scritte
- conversioni a unità interne o da unità interne
- se legge/scrive `Atoms` o `Trajectory`
- colonne/sezioni supportate

## 9. Checklist rapida per un agente che riprende il lavoro

1. Leggere questa mappa.
2. Controllare `git status` per distinguere codice versionato, modificato e untracked.
3. Se si lavora sui workflow fononici, partire da:
   - `src/isse/project_velocities.py`
   - `src/isse/phonon_temperatures.py`
   - `src/isse/helpers/cell_mapping.py`
4. Se si lavora su RDF/PBC, partire da:
   - `src/isse/radial_distribution.py`
   - `src/isse/helpers/periodic.py`
5. Se si lavora su I/O, partire da:
   - `src/isse/io/parse_lammps.py`
   - `src/isse/io/parse_gpumddump.py`
   - `src/isse/io/parse_vasp.py`
   - `src/isse/convert.py`
6. Prima di refactor importanti, introdurre test minimi su `Atoms`, `Trajectory`, parser, shape degli array, RDF/PBC e output di `calculate_temperature`.
