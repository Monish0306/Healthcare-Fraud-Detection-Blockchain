// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FraudDetection {

    struct FraudRecord {
        string  providerID;
        bool    isFraud;
        uint256 fraudProbability;   // stored as 0-10000 (multiply float by 10000)
        string  riskCategory;       // "Low", "Medium", "High"
        uint256 timestamp;
        bytes32 dataHash;           // keccak256 of providerID+probability+risk
    }

    mapping(uint256 => FraudRecord) public records;
    uint256 public recordCount;
    address public owner;

    event RecordStored(
        uint256 indexed recordId,
        string  providerID,
        bool    isFraud,
        string  riskCategory,
        uint256 timestamp
    );

    event RecordVerified(
        uint256 indexed recordId,
        bool    isValid
    );

    constructor() {
        owner       = msg.sender;
        recordCount = 0;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can store records");
        _;
    }

    function storeFraudRecord(
        string  memory _providerID,
        bool           _isFraud,
        uint256        _fraudProbability,
        string  memory _riskCategory,
        bytes32        _dataHash
    ) public onlyOwner returns (uint256) {

        records[recordCount] = FraudRecord({
            providerID      : _providerID,
            isFraud         : _isFraud,
            fraudProbability: _fraudProbability,
            riskCategory    : _riskCategory,
            timestamp       : block.timestamp,
            dataHash        : _dataHash
        });

        emit RecordStored(
            recordCount,
            _providerID,
            _isFraud,
            _riskCategory,
            block.timestamp
        );

        recordCount++;
        return recordCount - 1;
    }

    function getRecord(uint256 _recordId) public view returns (
        string  memory,
        bool,
        uint256,
        string  memory,
        uint256,
        bytes32
    ) {
        require(_recordId < recordCount, "Record does not exist");
        FraudRecord memory r = records[_recordId];
        return (
            r.providerID,
            r.isFraud,
            r.fraudProbability,
            r.riskCategory,
            r.timestamp,
            r.dataHash
        );
    }

    function verifyRecord(
        uint256 _recordId,
        bytes32 _expectedHash
    ) public returns (bool) {
        require(_recordId < recordCount, "Record does not exist");
        bool isValid = (records[_recordId].dataHash == _expectedHash);
        emit RecordVerified(_recordId, isValid);
        return isValid;
    }

    function getTotalRecords() public view returns (uint256) {
        return recordCount;
    }
}