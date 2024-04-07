from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import enum
import json
import threading
import time

from typing import Iterator
import psycopg
import pytest

ROOT_URL = "postgres:///postgres"
URL = "postgres:///pglockpy"
SET_UP_SQL = """
    CREATE TABLE t (id INT);
    CREATE TABLE u (id INT);
    CREATE TABLE v (with_unique_index INT UNIQUE);
    CREATE MATERIALIZED VIEW mat AS SELECT * FROM t;
    CREATE INDEX idx ON t (id);
    CREATE OR REPLACE FUNCTION f() RETURNS TRIGGER AS $$ BEGIN RETURN NEW; END; $$ LANGUAGE plpgsql;
    ALTER TABLE t ADD CONSTRAINT constr CHECK (id > 0) NOT VALID;
    CREATE SEQUENCE seq;
"""


@dataclass
class Connections:
    a: psycopg.Connection
    b: psycopg.Connection
    c: psycopg.Connection  # no implicit TRANSACTION


@dataclass(frozen=True)
class Lock:
    relation: str
    lock_kind: LockKind

    @staticmethod
    def from_mode(relation: str, mode: str) -> Lock:
        lock_kind = {
            "AccessExclusiveLock": L.ACCESS_EXCLUSIVE,
            "ExclusiveLock": L.EXCLUSIVE,
            "ShareRowExclusiveLock": L.SHARE_ROW_EXCLUSIVE,
            "ShareLock": L.SHARE,
            "ShareUpdateExclusiveLock": L.SHARE_UPDATE_EXCLUSIVE,
            "RowExclusiveLock": L.ROW_EXCLUSIVE,
            "RowShareLock": L.ROW_SHARE,
            "AccessShareLock": L.ACCESS_SHARE,
        }[mode]
        return Lock(relation, lock_kind)


@pytest.fixture
def conns() -> Iterator[Connections]:
    """Whole fresh database with N connections per test.

    Not quick, but simple.
    """
    try:
        with psycopg.connect(ROOT_URL, autocommit=True) as conn:
            conn.execute("DROP DATABASE pglockpy")
    except Exception:
        pass

    with psycopg.connect(ROOT_URL, autocommit=True) as conn:
        conn.execute("CREATE DATABASE pglockpy")

    with (
        psycopg.connect(URL) as a,
        psycopg.connect(URL) as b,
        psycopg.connect(URL, autocommit=True) as c,
    ):
        a.execute(SET_UP_SQL)
        a.commit()
        yield Connections(a, b, c)


class LockKind(enum.Enum):
    ACCESS_EXCLUSIVE = "ACCESS EXCLUSIVE"
    EXCLUSIVE = "EXCLUSIVE"
    SHARE_ROW_EXCLUSIVE = "SHARE ROW EXCLUSIVE"
    SHARE = "SHARE"
    SHARE_UPDATE_EXCLUSIVE = "SHARE UPDATE EXCLUSIVE"
    ROW_EXCLUSIVE = "ROW EXCLUSIVE"
    ROW_SHARE = "ROW SHARE"
    ACCESS_SHARE = "ACCESS SHARE"
    # SELECT ... FOR
    FOR_UPDATE = "FOR UPDATE"
    FOR_NO_KEY_UPDATE = "FOR NO KEY UPDATE"
    FOR_SHARE = "FOR SHARE"
    FOR_KEY_SHARE = "FOR KEY SHARE"


L = LockKind


class Statement(enum.Enum):
    DROP_TABLE = "DROP TABLE t"
    TRUNCATE = "TRUNCATE t"
    CREATE_TABLE = "CREATE TABLE v (id INT)"
    ALTER_TABLE = "ALTER TABLE t ADD COLUMN col INT"
    REINDEX = "REINDEX TABLE t"
    VACUUM_FULL = "VACUUM FULL"
    REFERESH_MATERIALIZED_VIEW = "REFRESH MATERIALIZED VIEW mat"
    ALTER_TABLE_FOREIGN_KEY = (
        "ALTER TABLE t ADD CONSTRAINT fk FOREIGN KEY (id) REFERENCES u (id)"
    )
    CREATE_TRIGGER = (
        "CREATE TRIGGER trig AFTER INSERT ON t FOR EACH ROW EXECUTE FUNCTION f()"
    )
    CREATE_INDEX = "CREATE INDEX idy ON t (id)"
    VACUUM = "VACUUM"
    ANALYZE = "ANALYZE"
    CREATE_INDEX_CONCURRENTLY = "CREATE INDEX CONCURRENTLY idy ON t (id)"
    CREATE_STATISTICS = "CREATE STATISTICS stat ON id FROM t"
    REINDEX_CONCURRENTLY = "REINDEX TABLE CONCURRENTLY t"
    ALTER_TABLE_SET_STATISTICS = "ALTER TABLE t ALTER COLUMN id SET STATISTICS 100"
    ALTER_TABLE_VALIDATE_CONSTRAINT = "ALTER TABLE t VALIDATE CONSTRAINT constr"
    ALTER_INDEX_RENAME = "ALTER INDEX idx RENAME TO idy"
    UPDATE = "UPDATE t SET id = 4"
    UPDATE_UNIQUE = "UPDATE v SET with_unique_index = 4"
    DELETE = "DELETE FROM t"
    INSERT = "INSERT INTO t VALUES (1)"
    MERGE = "MERGE INTO t USING u AS sub ON t.id = u.id WHEN MATCHED THEN DO NOTHING"
    SELECT_FOR_UPDATE = "SELECT * FROM t FOR UPDATE"
    SELECT_FOR_NO_KEY_UPDATE = "SELECT * FROM t FOR NO KEY UPDATE"
    SELECT_FOR_SHARE = "SELECT * FROM t FOR SHARE"
    SELECT_FOR_KEY_SHARE = "SELECT * FROM t FOR KEY SHARE"
    SELECT = "SELECT * FROM t"

    @property
    def name_no_underscore(self) -> str:
        return self.name.replace("_", " ")


@dataclass
class LockRelationship:
    original_lock: LockKind
    doesnt_block: list[LockKind]
    blocks: list[LockKind]


TABLE_LOCK_RELATIONSHIPS = [
    LockRelationship(
        original_lock=L.ACCESS_EXCLUSIVE,
        doesnt_block=[],
        blocks=      [L.ACCESS_SHARE, L.ROW_SHARE, L.ROW_EXCLUSIVE, L.SHARE_UPDATE_EXCLUSIVE, L.SHARE, L.SHARE_ROW_EXCLUSIVE, L.EXCLUSIVE, L.ACCESS_EXCLUSIVE],
    ),
    LockRelationship(
        original_lock=L.EXCLUSIVE,
        doesnt_block=[L.ACCESS_SHARE],
        blocks=      [                L.ROW_SHARE, L.ROW_EXCLUSIVE, L.SHARE_UPDATE_EXCLUSIVE, L.SHARE, L.SHARE_ROW_EXCLUSIVE, L.EXCLUSIVE, L.ACCESS_EXCLUSIVE],
    ),
    LockRelationship(
        original_lock=L.SHARE_ROW_EXCLUSIVE,
        doesnt_block=[L.ACCESS_SHARE, L.ROW_SHARE],
        blocks=      [                             L.ROW_EXCLUSIVE, L.SHARE_UPDATE_EXCLUSIVE, L.SHARE, L.SHARE_ROW_EXCLUSIVE, L.EXCLUSIVE, L.ACCESS_EXCLUSIVE],
    ),
    LockRelationship(
        original_lock=L.SHARE,
        doesnt_block=[L.ACCESS_SHARE, L.ROW_SHARE, L.SHARE],
        blocks=      [                             L.ROW_EXCLUSIVE, L.SHARE_UPDATE_EXCLUSIVE,          L.SHARE_ROW_EXCLUSIVE, L.EXCLUSIVE, L.ACCESS_EXCLUSIVE],
    ),
    LockRelationship(
        original_lock=L.SHARE_UPDATE_EXCLUSIVE,
        doesnt_block=[L.ACCESS_SHARE, L.ROW_SHARE, L.ROW_EXCLUSIVE],
        blocks=      [                                              L.SHARE_UPDATE_EXCLUSIVE, L.SHARE, L.SHARE_ROW_EXCLUSIVE, L.EXCLUSIVE, L.ACCESS_EXCLUSIVE],
    ),
    LockRelationship(
        original_lock=L.ROW_EXCLUSIVE,
        doesnt_block=[L.ACCESS_SHARE, L.ROW_SHARE, L.ROW_EXCLUSIVE, L.SHARE_UPDATE_EXCLUSIVE],
        blocks=      [                                                                        L.SHARE, L.SHARE_ROW_EXCLUSIVE, L.EXCLUSIVE, L.ACCESS_EXCLUSIVE],
    ),
    LockRelationship(
        original_lock=L.ROW_SHARE,
        doesnt_block=[L.ACCESS_SHARE, L.ROW_SHARE, L.ROW_EXCLUSIVE, L.SHARE_UPDATE_EXCLUSIVE, L.SHARE, L.SHARE_ROW_EXCLUSIVE],
        blocks=      [                                                                                                        L.EXCLUSIVE, L.ACCESS_EXCLUSIVE],
    ),
    LockRelationship(
        original_lock=L.ACCESS_SHARE,
        doesnt_block=[L.ACCESS_SHARE, L.ROW_SHARE, L.ROW_EXCLUSIVE, L.SHARE_UPDATE_EXCLUSIVE, L.SHARE, L.SHARE_ROW_EXCLUSIVE, L.EXCLUSIVE],
        blocks=      [                                                                                                                     L.ACCESS_EXCLUSIVE],
    ),
]  # fmt: skip

ROW_LOCK_RELATIONSHIPS = [
    LockRelationship(
        original_lock=L.FOR_UPDATE,
        doesnt_block=[],
        blocks=      [L.FOR_KEY_SHARE, L.FOR_SHARE, L.FOR_NO_KEY_UPDATE, L.FOR_UPDATE],
    ),
    LockRelationship(
        original_lock=L.FOR_NO_KEY_UPDATE,
        doesnt_block=[L.FOR_KEY_SHARE],
        blocks=      [                 L.FOR_SHARE, L.FOR_NO_KEY_UPDATE, L.FOR_UPDATE],
    ),
    LockRelationship(
        original_lock=L.FOR_SHARE,
        doesnt_block=[L.FOR_KEY_SHARE, L.FOR_SHARE],
        blocks=      [                              L.FOR_NO_KEY_UPDATE, L.FOR_UPDATE],
    ),
    LockRelationship(
        original_lock=L.FOR_KEY_SHARE,
        doesnt_block=[L.FOR_KEY_SHARE, L.FOR_SHARE, L.FOR_NO_KEY_UPDATE,],
        blocks=      [                                                   L.FOR_UPDATE],
    ),
]  # fmt: skip


STATEMENT_RELATION_LOCKS: list[tuple[Statement, set[Lock]]] = [
    (
        Statement.DROP_TABLE,
        {Lock(relation="t", lock_kind=L.ACCESS_EXCLUSIVE)},
    ),
    (
        Statement.TRUNCATE,
        {Lock(relation="t", lock_kind=L.ACCESS_EXCLUSIVE)},
    ),
    (
        Statement.CREATE_TABLE,
        set(),
    ),
    (
        Statement.VACUUM_FULL,
        {Lock(relation="u", lock_kind=L.ACCESS_EXCLUSIVE)},
    ),
    (
        Statement.ALTER_TABLE,
        {Lock(relation="t", lock_kind=L.ACCESS_EXCLUSIVE)},
    ),
    (
        Statement.REFERESH_MATERIALIZED_VIEW,
        {
            Lock(relation="t", lock_kind=L.ACCESS_SHARE),
            Lock(relation="mat", lock_kind=L.ACCESS_EXCLUSIVE),
            Lock(relation="mat", lock_kind=L.ACCESS_SHARE),
            Lock(relation="mat", lock_kind=L.EXCLUSIVE),
        },
    ),
    (
        Statement.REINDEX,
        {
            Lock(relation="t", lock_kind=L.SHARE)
            # This doesn't line up with my interpretation of the docs - it should take L.EXCLUSIVE
        },
    ),
    (
        Statement.ALTER_TABLE_FOREIGN_KEY,
        {
            Lock(relation="t", lock_kind=L.SHARE_ROW_EXCLUSIVE)
            # This doesn't line up with my interpretation of the docs - it should lock `u` too
        },
    ),
    (
        Statement.CREATE_TRIGGER,
        {Lock(relation="t", lock_kind=L.SHARE_ROW_EXCLUSIVE)},
    ),
    (
        Statement.CREATE_INDEX,
        {Lock(relation="t", lock_kind=L.SHARE)},
    ),
    (
        Statement.VACUUM,
        {Lock(relation="u", lock_kind=L.SHARE_UPDATE_EXCLUSIVE)},
    ),
    (
        Statement.ANALYZE,
        {Lock(relation="u", lock_kind=L.SHARE_UPDATE_EXCLUSIVE)},
    ),
    (
        Statement.CREATE_INDEX_CONCURRENTLY,
        {Lock(relation="t", lock_kind=L.SHARE_UPDATE_EXCLUSIVE)},
    ),
    (
        Statement.CREATE_STATISTICS,
        {Lock(relation="t", lock_kind=L.SHARE_UPDATE_EXCLUSIVE)},
    ),
    (
        Statement.REINDEX_CONCURRENTLY,
        {Lock(relation="t", lock_kind=L.SHARE_UPDATE_EXCLUSIVE)},
    ),
    (
        Statement.ALTER_TABLE_SET_STATISTICS,
        {Lock(relation="t", lock_kind=L.SHARE_UPDATE_EXCLUSIVE)},
    ),
    (
        Statement.ALTER_TABLE_VALIDATE_CONSTRAINT,
        {Lock(relation="t", lock_kind=L.SHARE_UPDATE_EXCLUSIVE)},
    ),
    (
        Statement.ALTER_INDEX_RENAME,
        set(
            # This doesn't line up with my interpretation of the docs - it should take L.SHARE_UPDATE_EXCLUSIVE
        ),
    ),
    (
        Statement.UPDATE,
        {Lock(relation="t", lock_kind=L.ROW_EXCLUSIVE)},
    ),
    (
        Statement.UPDATE_UNIQUE,
        {Lock(relation="v", lock_kind=L.ROW_EXCLUSIVE)},
    ),
    (
        Statement.DELETE,
        {Lock(relation="t", lock_kind=L.ROW_EXCLUSIVE)},
    ),
    (
        Statement.INSERT,
        {Lock(relation="t", lock_kind=L.ROW_EXCLUSIVE)},
    ),
    (
        Statement.MERGE,
        {Lock(relation="t", lock_kind=L.ROW_EXCLUSIVE)},
    ),
    (
        Statement.SELECT_FOR_UPDATE,
        {Lock(relation="t", lock_kind=L.ROW_SHARE)},
    ),
    (
        Statement.SELECT_FOR_NO_KEY_UPDATE,
        {Lock(relation="t", lock_kind=L.ROW_SHARE)},
    ),
    (
        Statement.SELECT_FOR_SHARE,
        {Lock(relation="t", lock_kind=L.ROW_SHARE)},
    ),
    (
        Statement.SELECT_FOR_KEY_SHARE,
        {Lock(relation="t", lock_kind=L.ROW_SHARE)},
    ),
    (
        Statement.SELECT,
        {Lock(relation="t", lock_kind=L.ACCESS_SHARE)},
    ),
]


@pytest.mark.parametrize(
    "original_lock, doesnt_block, blocks",
    [(r.original_lock, r.doesnt_block, r.blocks) for r in TABLE_LOCK_RELATIONSHIPS],
)
def test_which_locks_block(
    conns: Connections,
    original_lock: L,
    doesnt_block: list[L],
    blocks: list[L],
) -> None:
    with conns.a.cursor() as cursor_a, conns.b.cursor() as cursor_b:
        cursor_a.execute(f"LOCK TABLE ONLY t IN {original_lock.value} MODE NOWAIT")

        for lock_kind in doesnt_block:
            cursor_b.execute(f"LOCK TABLE ONLY t IN {lock_kind.value} MODE NOWAIT")
            conns.b.rollback()

        for lock_kind in blocks:
            with pytest.raises(psycopg.errors.LockNotAvailable):
                cursor_b.execute(f"LOCK TABLE ONLY t IN {lock_kind.value} MODE NOWAIT")
            conns.b.rollback()


@pytest.mark.parametrize(
    "original_lock, doesnt_block, blocks",
    [(r.original_lock, r.doesnt_block, r.blocks) for r in ROW_LOCK_RELATIONSHIPS],
)  # fmt: skip
def test_for_update_locks(
    conns: Connections,
    original_lock: L,
    doesnt_block: list[L],
    blocks: list[L],
) -> None:
    with conns.a.cursor() as cursor_a, conns.b.cursor() as cursor_b:
        cursor_a.execute("INSERT INTO t VALUES (1)")
        conns.a.commit()

        cursor_a.execute(f"SELECT * FROM t {original_lock.value} NOWAIT")

        # Note, we can always do a standard SELECT
        cursor_b.execute("SELECT * FROM t")

        for lock_kind in doesnt_block:
            cursor_b.execute(f"SELECT * FROM t {lock_kind.value} NOWAIT")
            conns.b.rollback()

        for lock_kind in blocks:
            with pytest.raises(psycopg.errors.LockNotAvailable):
                cursor_b.execute(f"SELECT * FROM t {lock_kind.value} NOWAIT")
            conns.b.rollback()


@pytest.mark.parametrize("statement, locks", STATEMENT_RELATION_LOCKS)
def test_statement_takes_locks(
    conns: Connections, statement: Statement, locks: set[Lock]
) -> None:
    with conns.a.cursor() as lock_cursor:
        lock_tables(lock_cursor, ["t", "u", "v"])

        with execute_in_thread(conns.c, statement.value):
            pid = conns.c.info.backend_pid
            wait_for_statement(lock_cursor, pid)
            actual_locks = get_all_locks(lock_cursor, pid)

        assert actual_locks == locks


def test_update_locks(conns: Connections) -> None:
    with conns.a.cursor() as cursor_a, conns.b.cursor() as cursor_b:
        cursor_a.execute("INSERT INTO t VALUES (1)")
        conns.a.commit()

        cursor_a.execute(f"UPDATE t SET id = 2")

        # Note, we can always do a standard SELECT
        cursor_b.execute("SELECT * FROM t")

        for lock_kind in [
            L.FOR_KEY_SHARE,
        ]:
            cursor_b.execute(f"SELECT * FROM t {lock_kind.value} NOWAIT")
            conns.b.rollback()

        for lock_kind in [
            L.FOR_SHARE,
            L.FOR_NO_KEY_UPDATE,
            L.FOR_UPDATE,
        ]:
            with pytest.raises(psycopg.errors.LockNotAvailable):
                cursor_b.execute(f"SELECT * FROM t {lock_kind.value} NOWAIT")
            conns.b.rollback()


def test_update_unique_locks(conns: Connections) -> None:
    with conns.a.cursor() as cursor_a, conns.b.cursor() as cursor_b:
        cursor_a.execute("INSERT INTO v VALUES (1)")
        conns.a.commit()

        cursor_a.execute(f"UPDATE v SET with_unique_index = 2")

        # Note, we can always do a standard SELECT
        cursor_b.execute("SELECT * FROM v")

        for lock_kind in [
            L.FOR_KEY_SHARE,
            L.FOR_SHARE,
            L.FOR_NO_KEY_UPDATE,
            L.FOR_UPDATE,
        ]:
            with pytest.raises(psycopg.errors.LockNotAvailable):
                cursor_b.execute(f"SELECT * FROM v {lock_kind.value} NOWAIT")
            conns.b.rollback()


def test_delete_locks(conns: Connections) -> None:
    with conns.a.cursor() as cursor_a, conns.b.cursor() as cursor_b:
        cursor_a.execute("INSERT INTO t VALUES (1)")
        conns.a.commit()

        cursor_a.execute(f"DELETE FROM t")

        # Note, we can always do a standard SELECT
        cursor_b.execute("SELECT * FROM t")

        for lock_kind in [
            L.FOR_KEY_SHARE,
            L.FOR_SHARE,
            L.FOR_NO_KEY_UPDATE,
            L.FOR_UPDATE,
        ]:
            with pytest.raises(psycopg.errors.LockNotAvailable):
                cursor_b.execute(f"SELECT * FROM t {lock_kind.value} NOWAIT")
            conns.b.rollback()


# Helpers


@contextmanager
def execute_in_thread(conn: psycopg.Connection, statement: str) -> Iterator[None]:
    thread = threading.Thread(target=_execute, args=(conn, statement), daemon=True)
    thread.start()
    yield
    conn.cancel()
    thread.join()


def _execute(conn: psycopg.Connection, statement: str) -> None:
    try:
        conn.execute(statement)
    except psycopg.errors.QueryCanceled:
        pass


def lock_tables(cursor: psycopg.Cursor, table_names: list[str]) -> None:
    for table_name in table_names:
        cursor.execute(
            f'LOCK TABLE ONLY "{table_name}" IN ACCESS EXCLUSIVE MODE NOWAIT'
        )


def wait_for_statement(cursor: psycopg.Cursor, pid: int) -> None:
    qry = """
        SELECT 1
        FROM pg_stat_activity
        WHERE pid = %s AND query != 'BEGIN'
    """
    while True:
        cursor.execute(qry, [pid])
        if cursor.fetchall():
            break


def get_all_locks(cursor: psycopg.Cursor, pid: int) -> set[Lock]:
    qry = """
        SELECT (pg_locks.relation)::regclass, pg_locks.mode
        FROM pg_locks
        JOIN pg_class ON pg_class.oid = pg_locks.relation
        WHERE pg_locks.pid = %s
    """
    cursor.execute(qry, [pid])
    return {
        Lock.from_mode(relation=relation_name, mode=mode)
        for [relation_name, mode] in cursor.fetchall()
    }


# Dump JSON


def test_dump_json() -> None:
    data = dict(
        statements={
            v.name_no_underscore: dict(example=v.value, lockTypes=[]) for v in Statement
        },
        locks={
            r.original_lock.value: dict(
                kind="TABLE",
                blocks=[x.value for x in r.blocks],
                doesntBlock=[x.value for x in r.doesnt_block],
                statements=[],
            )
            for r in TABLE_LOCK_RELATIONSHIPS
        }
        | {
            r.original_lock.value: dict(
                kind="ROW",
                blocks=[x.value for x in r.blocks],
                doesntBlock=[x.value for x in r.doesnt_block],
                statements=[],
            )
            for r in ROW_LOCK_RELATIONSHIPS
        },
    )
    for statement, locks in STATEMENT_RELATION_LOCKS:
        for l in locks:
            if l.lock_kind.value not in data["statements"][statement.name_no_underscore]["lockTypes"]:
                data["statements"][statement.name_no_underscore]["lockTypes"].append(
                    l.lock_kind.value
                )
            if statement.name_no_underscore not in data["locks"][l.lock_kind.value]["statements"]:
                data["locks"][l.lock_kind.value]["statements"].append(
                    statement.name_no_underscore
                )

    # Add assumed row locks
    for statement, lock_kind in [
        (Statement.SELECT_FOR_UPDATE, L.FOR_UPDATE),
        (Statement.SELECT_FOR_NO_KEY_UPDATE, L.FOR_NO_KEY_UPDATE),
        (Statement.SELECT_FOR_SHARE, L.FOR_SHARE),
        (Statement.SELECT_FOR_KEY_SHARE, L.FOR_KEY_SHARE),
        (Statement.UPDATE, L.FOR_NO_KEY_UPDATE),
        (Statement.UPDATE_UNIQUE, L.FOR_UPDATE),
        (Statement.DELETE, L.FOR_UPDATE),
    ]:
        data["statements"][statement.name_no_underscore]["lockTypes"].append(
            lock_kind.value
        )
        data["locks"][lock_kind.value]["statements"].append(
            statement.name_no_underscore
        )

    with open("pglockpy.json", "w") as f:
        json.dump(data, f, indent=4)
