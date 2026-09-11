"""
Title: Step-Back Prompting (OpenRouter)
Step-back prompting asks the model to first work out the general
principles behind a question, then apply those principles to answer
the specific question.

uv run python src/10_step_back_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Step 1: Extract high-level security principles
    step_back_prompt = ChatPromptTemplate.from_template("""
        You are an Android application security engineer skilled at
        identifying underlying security principles.

        Given this specific question:

        {question}

        What are the key security principles, concepts, or frameworks
        needed to answer this correctly? List 2-3 fundamental principles,
        then briefly explain each.

        PRINCIPLES:
    """)

    # Step 2: Apply the principles to the specific question
    apply_prompt = ChatPromptTemplate.from_template("""
        Use the following principles to answer the specific question.

        PRINCIPLES:
        {principles}

        SPECIFIC QUESTION:
        {question}

        Apply the principles step-by-step to arrive at the answer.

        ANSWER:
    """)

    # The two chains that make up the step-back pipeline
    step_back_chain = step_back_prompt | llm | parser
    apply_chain = apply_prompt | llm | parser

    # Sample security questions
    questions = [
        """
        An Android app stores its OAuth access token in SharedPreferences
        without encryption. What risks does this expose, and how should
        it be stored instead?
        """,

        """
        An Android app's WebView component has addJavascriptInterface()
        enabled and loads untrusted URLs. What vulnerability class does
        this expose, and how should it be mitigated?
        """,

        """
        An Android app accepts any TLS certificate for its backend API
        calls, with no certificate pinning. What risks does this expose,
        and how should it be fixed?
        """,
    ]

    # Invoke the step-back pipeline for each question, one at a time
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"QUESTION {i}")
        print("=" * 60)
        print(f"\n{question.strip()}\n")
        print("-" * 60)

        # Show the step-back process
        principles = step_back_chain.invoke({"question": question})
        print("STEP-BACK PRINCIPLES:")
        print(principles)
        print("\n" + "-" * 60)

        # Apply the principles to reach the final answer
        answer = apply_chain.invoke({"question": question, "principles": principles})
        print("FINAL ANSWER:")
        print(answer)


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------



============================================================
QUESTION 1
============================================================

An Android app stores its OAuth access token in SharedPreferences
        without encryption. What risks does this expose, and how should
        it be stored instead?

------------------------------------------------------------
STEP-BACK PRINCIPLES:
Certainly! Here are three fundamental security principles that are key to addressing the question:

1. **Data Encryption**:
   - **Explanation**: Data encryption is a critical principle in securing sensitive information. When data is encrypted, it is converted into a format that cannot be easily understood by unauthorized parties. In the context of storing OAuth access tokens in SharedPreferences, encrypting the token ensures that even if an attacker gains access to the device's file system or SharedPreferences, they will not be able to read the token without the encryption key.

2. **Secure Storage Mechanisms**:
   - **Explanation**: Secure storage mechanisms ensure that sensitive data is stored in a way that minimizes the risk of unauthorized access. For Android apps, this often involves using built-in secure storage options provided by the platform, such as `KeyStore` or `EncryptedSharedPreferences`. These mechanisms provide a higher level of security compared to plain text storage because they use strong cryptographic algorithms and protect against common attacks like tampering and side-channel attacks.

3. **Least Privilege Principle**:
   - **Explanation**: The least privilege principle states that a user or application should have only the permissions necessary to perform its tasks. In the context of storing OAuth access tokens, this means that the app should store the token with the minimum necessary permissions and access controls. For example, the token should not be accessible to other apps on the same device unless absolutely required, and it should be stored in a way that limits the scope of potential damage if the token is compromised.

By adhering to these principles, you can significantly reduce the risks associated with storing sensitive information like OAuth access tokens in SharedPreferences.

------------------------------------------------------------
FINAL ANSWER:
### Step-by-Step Answer Using the Principles

#### 1. **Identify the Risks**

When an Android app stores its OAuth access token in SharedPreferences without encryption, several risks are exposed:

- **Data Exposure**: If an attacker gains access to the device's file system or SharedPreferences, they can read the OAuth access token directly.
- **Security Breach**: The token is not protected by any encryption, making it vulnerable to unauthorized use and potential misuse.
- **Unauthorized Access**: Without encryption, the token can be easily accessed by other apps on the same device if they have the necessary permissions.

#### 2. **Apply Data Encryption Principle**

To mitigate these risks, the OAuth access token should be encrypted before being stored in SharedPreferences. Here’s how you can implement this:

- **Encryption Process**:
  - Use a strong encryption algorithm (e.g., AES) to encrypt the OAuth access token.
  - Store the encrypted token in SharedPreferences.
  - Ensure that the encryption key is securely managed and not stored in plain text.

Example code for encryption and decryption using AES in Java:

```java
import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.util.Base64;

public class EncryptionUtil {
    private static final String ALGORITHM = "AES";
    private static final String KEY = "YourSecretKey123"; // Must be 16 bytes long

    public static String encrypt(String value) throws Exception {
        SecretKeySpec secretKey = new SecretKeySpec(KEY.getBytes(), ALGORITHM);
        Cipher cipher = Cipher.getInstance(ALGORITHM);
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encrypted = cipher.doFinal(value.getBytes());
        return Base64.getEncoder().encodeToString(encrypted);
    }

    public static String decrypt(String encryptedValue) throws Exception {
        SecretKeySpec secretKey = new SecretKeySpec(KEY.getBytes(), ALGORITHM);
        Cipher cipher = Cipher.getInstance(ALGORITHM);
        cipher.init(Cipher.DECRYPT_MODE, secretKey);
        byte[] decodedValue = Base64.getDecoder().decode(encryptedValue);
        byte[] decrypted = cipher.doFinal(decodedValue);
        return new String(decrypted);
    }
}
```

#### 3. **Apply Secure Storage Mechanisms Principle**

To further enhance security, consider using secure storage mechanisms provided by Android, such as `EncryptedSharedPreferences` or `KeyStore`.

- **EncryptedSharedPreferences Example**:
  - Use `EncryptedSharedPreferences` to store the encrypted OAuth access token.
  - This ensures that even if the device is rooted or compromised, the data remains secure.

Example code for using `EncryptedSharedPreferences`:

```java
import android.content.Context;
import androidx.security.crypto.EncryptedSharedPreferences;
import androidx.security.crypto.MasterKeys;

public class SecureStorageManager {
    private static final String PREFERENCES_NAME = "SecurePrefs";

    public static EncryptedSharedPreferences getEncryptedSharedPreferences(Context context) throws Exception {
        String masterKeyAlias = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC);
        return EncryptedSharedPreferences.create(
                PREFERENCES_NAME,
                masterKeyAlias,
                context,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        );
    }

    public static void saveOAuthToken(Context context, String token) throws Exception {
        EncryptedSharedPreferences sharedPreferences = getEncryptedSharedPreferences(context);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString("oauth_token", token);
        editor.apply();
    }

    public static String loadOAuthToken(Context context) throws Exception {
        EncryptedSharedPreferences sharedPreferences = getEncryptedSharedPreferences(context);
        return sharedPreferences.getString("oauth_token", "");
    }
}
```

#### 4. **Apply Least Privilege Principle**

Ensure that the OAuth access token is stored with the minimum necessary permissions and access controls:

- **Access Controls**: Limit the access to the SharedPreferences or `EncryptedSharedPreferences` to only the components of your app that need it.
- **Permissions**: Ensure that no other app on the device can access the SharedPreferences or `EncryptedSharedPreferences`.

By following these steps, you can significantly reduce the risks associated with storing sensitive information like OAuth access tokens in SharedPreferences.

============================================================
QUESTION 2
============================================================

An Android app's WebView component has addJavascriptInterface()
        enabled and loads untrusted URLs. What vulnerability class does
        this expose, and how should it be mitigated?

------------------------------------------------------------
STEP-BACK PRINCIPLES:
Certainly! Here are three fundamental security principles that are key to understanding and mitigating the vulnerability in this scenario:

### PRINCIPLES:

1. **Least Privilege (Principle of Least Privilege, PoLP)**
2. **Separation of Concerns**
3. **Input Validation and Sanitization**

### EXPLANATION:

1. **Least Privilege (PoLP):**
   - **Explanation:** The principle of least privilege is a fundamental security concept that suggests that users and processes should be granted only the minimum level of access necessary to perform their tasks. In the context of the WebView component, this means that any JavaScript interfaces exposed via `addJavascriptInterface()` should have the minimum permissions required to perform their intended functions. By limiting the permissions, you reduce the potential damage if an attacker gains control over these interfaces.
   - **Application to the Scenario:** Ensure that the methods exposed through `addJavascriptInterface()` have the least amount of privileges needed to interact with the app's functionality. Avoid exposing sensitive operations or data.

2. **Separation of Concerns:**
   - **Explanation:** This principle involves dividing a system into distinct components or modules, each responsible for a specific aspect of the system. By separating concerns, you can isolate vulnerabilities and limit the impact of any security breach. In the case of WebView, this means keeping trusted and untrusted content strictly separated.
   - **Application to the Scenario:** Ensure that the WebView component does not load untrusted URLs directly. Instead, use a proxy or a secure gateway to validate and sanitize the URLs before loading them. This helps prevent the execution of malicious scripts that could exploit the `addJavascriptInterface()` method.

3. **Input Validation and Sanitization:**
   - **Explanation:** Input validation and sanitization involve checking and cleaning user inputs to ensure they meet expected formats and do not contain harmful content. This is crucial for preventing injection attacks and other forms of exploitation.
   - **Application to the Scenario:** Validate and sanitize all inputs received by the `addJavascriptInterface()` methods. Ensure that the inputs are safe and do not contain any malicious code. Additionally, consider using Content Security Policy (CSP) headers to further restrict the sources from which scripts can be loaded.

By adhering to these principles, you can significantly mitigate the risks associated with enabling `addJavascriptInterface()` on a WebView that loads untrusted URLs.

------------------------------------------------------------
FINAL ANSWER:
### Vulnerability Class and Mitigation

#### Step 1: Identify the Vulnerability Class
The scenario where an Android app's WebView component has `addJavascriptInterface()` enabled and loads untrusted URLs exposes a **Cross-Site Scripting (XSS) vulnerability**. XSS vulnerabilities occur when an attacker can inject malicious scripts into content viewed by other users, leading to potential data theft, session hijacking, or other malicious activities.

#### Step 2: Apply the Principle of Least Privilege (PoLP)
1. **Explanation:** The principle of least privilege suggests that any JavaScript interfaces exposed via `addJavascriptInterface()` should have the minimum level of permissions necessary to perform their intended functions.
2. **Application to the Scenario:**
   - Ensure that the methods exposed through `addJavascriptInterface()` have the least amount of privileges needed to interact with the app's functionality.
   - Avoid exposing sensitive operations or data through these interfaces.
   - For example, if you need to expose a method to handle user input, ensure it only performs the specific task required and does not have broader access to the app's resources.

#### Step 3: Apply the Separation of Concerns Principle
1. **Explanation:** This principle involves dividing a system into distinct components or modules, each responsible for a specific aspect of the system. By separating concerns, you can isolate vulnerabilities and limit the impact of any security breach.
2. **Application to the Scenario:**
   - Ensure that the WebView component does not load untrusted URLs directly.
   - Use a proxy or a secure gateway to validate and sanitize the URLs before loading them.
   - For example, implement a server-side validation mechanism to check the URLs before they are loaded in the WebView.
   - Consider using Content Security Policy (CSP) headers to further restrict the sources from which scripts can be loaded.

#### Step 4: Apply Input Validation and Sanitization
1. **Explanation:** Input validation and sanitization involve checking and cleaning user inputs to ensure they meet expected formats and do not contain harmful content.
2. **Application to the Scenario:**
   - Validate and sanitize all inputs received by the `addJavascriptInterface()` methods.
   - Ensure that the inputs are safe and do not contain any malicious code.
   - Implement input validation on both the client and server sides to prevent injection attacks.
   - Use libraries or frameworks that provide built-in protection against common injection attacks.

### Summary
By adhering to the principles of least privilege, separation of concerns, and input validation and sanitization, you can mitigate the risks associated with enabling `addJavascriptInterface()` on a WebView that loads untrusted URLs. Specifically:

- **Least Privilege (PoLP):** Ensure that the methods exposed through `addJavascriptInterface()` have the minimum necessary permissions.
- **Separation of Concerns:** Use a proxy or secure gateway to validate and sanitize URLs before loading them in the WebView.
- **Input Validation and Sanitization:** Validate and sanitize all inputs received by the `addJavascriptInterface()` methods to prevent injection attacks.

These steps will help protect your application from XSS and other related vulnerabilities.

============================================================
QUESTION 3
============================================================

An Android app accepts any TLS certificate for its backend API
        calls, with no certificate pinning. What risks does this expose,
        and how should it be fixed?

------------------------------------------------------------
STEP-BACK PRINCIPLES:
Certainly! Here are three key security principles that are essential to understanding and addressing the issue of accepting any TLS certificate for backend API calls without certificate pinning:

1. **Certificate Pinning**:
   - **Explanation**: Certificate pinning is a security practice where an application verifies the server's SSL/TLS certificate against a pre-defined public key or certificate hash (pin) instead of relying on the system's trust store. This helps prevent man-in-the-middle (MITM) attacks, where an attacker could intercept and modify communications between the client and server by presenting a fraudulent certificate.

2. **Transport Layer Security (TLS)**:
   - **Explanation**: TLS is a cryptographic protocol designed to provide communication security over a computer network. It ensures data confidentiality, integrity, and authenticity. By accepting any TLS certificate, the app is vulnerable to MITM attacks, as it does not verify the identity of the server beyond what the operating system's trust store provides.

3. **Least Privilege and Defense in Depth**:
   - **Explanation**: The principle of least privilege means that applications should only have the permissions necessary to perform their intended functions. Defense in depth involves implementing multiple layers of security controls to protect against various types of threats. In this context, not using certificate pinning leaves the app vulnerable to attacks that could be mitigated by additional security measures like certificate pinning.

These principles highlight the importance of verifying the server's identity and ensuring that security measures are robust and layered to protect against potential vulnerabilities.

------------------------------------------------------------
FINAL ANSWER:
To address the specific question, we will apply the security principles step-by-step to understand the risks and propose a solution.

### Step 1: Identify the Risks

Given that the Android app accepts any TLS certificate for its backend API calls without certificate pinning, we can identify the following risks:

1. **Man-in-the-Middle (MITM) Attacks**:
   - **Explanation**: Without certificate pinning, the app relies solely on the system's trust store to validate the server's certificate. An attacker could intercept and modify communications by presenting a fraudulent certificate that is trusted by the system's trust store. This allows the attacker to eavesdrop on or tamper with the data being exchanged between the client and server.

2. **Lack of Identity Verification**:
   - **Explanation**: The app does not verify the server's identity beyond what the operating system's trust store provides. This means that if the trust store is compromised or contains a fraudulent certificate, the app will still consider it valid, leading to potential security breaches.

3. **Inadequate Security Practices**:
   - **Explanation**: Accepting any TLS certificate without certificate pinning is a poor security practice. It leaves the app vulnerable to various types of attacks and does not provide the necessary level of assurance required for secure communication.

### Step 2: Apply Certificate Pinning

To mitigate these risks, we need to implement certificate pinning. Here’s how to do it:

1. **Implement Certificate Pinning**:
   - **Explanation**: Certificate pinning involves verifying the server's SSL/TLS certificate against a pre-defined public key or certificate hash instead of relying on the system's trust store. This ensures that even if the system's trust store is compromised, the app can still detect and reject fraudulent certificates.
   - **Implementation Steps**:
     - Obtain the public key or certificate hash from the server.
     - Store this information securely in the app.
     - During each API call, verify the server's certificate against the stored public key or certificate hash.
     - If the certificate does not match, terminate the connection and display an error message to the user.

2. **Ensure Least Privilege and Defense in Depth**:
   - **Explanation**: Implement additional security measures to ensure that the app has the least privilege necessary and that there are multiple layers of security controls.
   - **Implementation Steps**:
     - Use secure network protocols and encryption.
     - Implement other security practices such as input validation, output encoding, and secure storage of sensitive data.
     - Regularly update and patch the app to address known vulnerabilities.

### Step 3: Conclusion

By implementing certificate pinning and adhering to the principle of least privilege and defense in depth, the Android app can significantly reduce its exposure to MITM attacks and other security risks. This approach ensures that the app verifies the server's identity more robustly and provides a higher level of security for the communication between the client and server.

In summary, the risks exposed by accepting any TLS certificate without certificate pinning include the vulnerability to MITM attacks and inadequate identity verification. To fix these issues, the app should implement certificate pinning and follow best security practices such as least privilege and defense in depth.


"""