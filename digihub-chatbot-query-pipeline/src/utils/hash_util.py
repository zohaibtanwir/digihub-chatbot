import hashlib

class HashUtils:
    @staticmethod
    def hash_user_id(user_id: str) -> str:
        """
        Hash the userId with sha256.

        Args:
            user_id (str): The user ID to be hashed.

        Returns:
            str: The hashed user ID.
        """

        # Create a sha256 hash object and update it with the user ID encoded to bytes
        sha256_hash = hashlib.sha256(user_id.encode('utf-8'))

         # Get the hexadecimal representation of the hash
        hashed_user_id = sha256_hash.hexdigest()

        return hashed_user_id
