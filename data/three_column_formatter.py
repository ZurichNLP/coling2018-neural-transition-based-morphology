import glob
import csv

celex_path  = '/mnt/storage/hex/users/makarov/celex-by-task/*/0500/*/*'
lemmat_path = '/mnt/storage/hex/users/makarov/wicentowski-lemmatization/*/*/*'

for path in (celex_path, lemmat_path):
    fns = glob.glob(path)
    for fn in fns:
        with open(fn) as f:
            out_rows = []
            for row in csv.reader(f, delimiter='\t'):
                assert len(row) == 2, (row, fn)
                # --no-feat-format
                # no --sigm2017format flag!
                new_row = row[0], '', row[1]
                out_rows.append(new_row)

        with open(fn, 'w') as w:
            csv.writer(w, delimiter='\t').writerows(out_rows)
