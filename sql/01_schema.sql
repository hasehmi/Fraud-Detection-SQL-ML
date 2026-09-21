-- Fraud Detection Database Schema
-- Run this first to create the tables, then load the CSVs from data/

DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS merchants;

CREATE TABLE customers (
    customer_id     INTEGER PRIMARY KEY,
    home_city       VARCHAR(50) NOT NULL,
    avg_spend       NUMERIC(10, 2) NOT NULL,
    favorite_category VARCHAR(50) NOT NULL,
    risk_profile    VARCHAR(10) NOT NULL
);

CREATE TABLE merchants (
    merchant_id       INTEGER PRIMARY KEY,
    merchant_category VARCHAR(50) NOT NULL,
    city              VARCHAR(50) NOT NULL
);

CREATE TABLE transactions (
    txn_id          INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    merchant_id     INTEGER NOT NULL REFERENCES merchants(merchant_id),
    amount          NUMERIC(10, 2) NOT NULL,
    timestamp       TIMESTAMP NOT NULL,
    city            VARCHAR(50) NOT NULL,
    is_fraud        SMALLINT NOT NULL
);

CREATE INDEX idx_txn_customer_time ON transactions (customer_id, timestamp);
CREATE INDEX idx_txn_merchant ON transactions (merchant_id);
