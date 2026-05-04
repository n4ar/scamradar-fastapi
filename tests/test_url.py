from app.pipelines.url import check_domain_impersonation, extract_domain

def test_extract_domain():
    assert extract_domain("http://kbank-th.xyz/login") == "kbank-th.xyz"
    assert extract_domain("https://www.scb-reward.xyz/claim") == "scb-reward.xyz"

def test_impersonation_detected():
    brand, detected = check_domain_impersonation("http://kbank-bonus.xyz/claim")
    assert detected is True
    assert brand == "กสิกรไทย"

def test_legit_domain_not_flagged():
    brand, detected = check_domain_impersonation("https://kbank.co.th/login")
    assert detected is False

def test_scb_impersonation():
    brand, detected = check_domain_impersonation("http://scb-reward.xyz")
    assert detected is True
    assert brand == "ไทยพาณิชย์"

def test_unrelated_domain_not_flagged():
    brand, detected = check_domain_impersonation("https://example.com")
    assert detected is False
