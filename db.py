import psycopg2
from contextlib import contextmanager
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from credentials import user, dbname, host, port, pswd

from pdb import set_trace as pp

class psqlServer():
    def __init__(self):
        self.user = user
        self.dbname = dbname
        self.port = port
        self.pswd = pswd
        self.pool = ThreadedConnectionPool(1,20, host=host, port=port, database=dbname, user=user, password=pswd)

    @contextmanager
    def get_conn(self):
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def get_curs(self, commit=False):
        with self.get_conn() as conn:
            curs = conn.cursor(cursor_factory=RealDictCursor)
            try:
                yield curs

            finally:
                curs.close()


    def execute(self, statement, *args):
        with self.get_curs() as curs:
            curs.execute(statement, args)
            return curs.fetchall()

    def execute_insert(self, statement, *args):
        with self.get_conn() as conn:
            curs = conn.cursor()
            curs.execute(statement, args)
            conn.commit()
            curs.close()
            if curs.description is not None:
                return curs.fetchall()

def test():
    sv = psqlServer()
    print(sv.execute('SELECT 1;'))
