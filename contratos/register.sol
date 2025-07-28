// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TextRegistry {
    // Mapeamento que associa um hash/texto a um endereço de quem o registrou
    mapping(string => address) private registeredBy;

    // Mapeamento que associa um hash/texto a um carimbo de tempo (timestamp)
    mapping(string => uint256) private registeredAt;

    // Evento que será emitido toda vez que um texto/hash for registrado
    event TextRegistered(string indexed _textOrHash, address indexed _registrant, uint256 _timestamp);

    
    function registerText(string memory _textOrHash) public {
        require(bytes(_textOrHash).length > 0, "O texto ou hash nao pode ser vazio.");
        require(registeredAt[_textOrHash] == 0, "Este texto ou hash ja foi registrado.");

        registeredBy[_textOrHash] = msg.sender;
        registeredAt[_textOrHash] = block.timestamp;

        emit TextRegistered(_textOrHash, msg.sender, block.timestamp);
    }

   
    function getRegistrant(string memory _textOrHash) public view returns (address) {
        return registeredBy[_textOrHash];
    }

    
    function getRegistrationTimestamp(string memory _textOrHash) public view returns (uint256) {
        return registeredAt[_textOrHash];
    }
}
