"""
Title: Skeleton of Thought Prompting (OpenRouter)

Skeleton of thought writes a short outline first, then expands each section
of the outline on its own, and finally joins the sections into one article.
Planning the structure first keeps long writing organized.

uv run python src/18_skeleton_of_thought_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def skeleton_of_thought(task: str, skeleton_chain, expand_chain, combine_chain) -> str:
    """Generate long-form content using the skeleton-first approach."""

    # Step 1: Generate the skeleton
    skeleton = skeleton_chain.invoke({"task": task})
    print("SKELETON:")
    print(skeleton)
    print("\n" + "=" * 60 + "\n")

    # Parse the sections from the skeleton (lines that start with a number)
    sections = [
        line.strip() for line in skeleton.split("\n") if line.strip() and line.strip()[0].isdigit()
    ]

    # Step 2: Expand each section
    expanded = []
    for section in sections:
        print(f"Expanding: {section[:50]}...")
        content = expand_chain.invoke({"task": task, "skeleton": skeleton, "section": section})
        expanded.append(f"## {section}\n\n{content}")

    print("\n" + "=" * 60 + "\n")

    # Step 3: Combine the sections
    final = combine_chain.invoke({"task": task, "expanded_sections": "\n\n---\n\n".join(expanded)})

    return final


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct", 
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Step 1 prompt: generate the skeleton
    skeleton_prompt = ChatPromptTemplate.from_template("""
        Create a structured outline for the following writing task.
        Include 4-6 main sections with brief descriptions of what each should cover.

        TASK: {task}

        OUTLINE:
        1. [Section Title]: [Brief description of content]
        2. [Section Title]: [Brief description of content]
        ...
    """)

    # Step 2 prompt: expand one section
    expand_prompt = ChatPromptTemplate.from_template("""
        Write the content for this section of an article.

        ARTICLE TOPIC: {task}

        FULL OUTLINE:
        {skeleton}

        SECTION TO EXPAND: {section}

        Write 2-3 paragraphs for this section only. Be specific and substantive.
    """)

    # Step 3 prompt: combine the sections
    combine_prompt = ChatPromptTemplate.from_template("""
        Combine these sections into a cohesive article with smooth transitions.

        TOPIC: {task}

        SECTIONS:
        {expanded_sections}

        Create the final article with an introduction and conclusion.
        Ensure smooth transitions between sections.
    """)

    # Create the three chains
    skeleton_chain = skeleton_prompt | llm | parser
    expand_chain = expand_prompt | llm | parser
    combine_chain = combine_prompt | llm | parser

    # A long-form writing task
    task = (
        "Write an article explaining how an Android app keeps its network "
        "traffic safe with HTTPS and certificate pinning, for a developer audience"
    )

    print(f"TASK: {task}")
    print("=" * 60 + "\n")

    article = skeleton_of_thought(task, skeleton_chain, expand_chain, combine_chain)
    print("FINAL ARTICLE:")
    print(article)


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


TASK: Write an article explaining how an Android app keeps its network traffic safe with HTTPS and certificate pinning, for a developer audience
============================================================

SKELETON:
1. The Threat Landscape: Why Network Security Matters: Introduce the risk of Man-in-the-Middle (MitM) attacks on Android apps and establish the baseline requirement of securing data in transit for a developer audience.

2. Establishing the Baseline with HTTPS and TLS: Explain how HTTPS (TLS/SSL) encrypts traffic and how Android enforces it by default, including the role of the Network Security Configuration file for customizing baseline security rules and handling cleartext traffic.

3. The Limitations of HTTPS and the Case for Pinning: Detail why standard HTTPS is vulnerable to compromised Certificate Authorities (CAs) or corporate proxy setups, and explain how certificate pinning mitigates these risks by trusting only specific known certificates or public keys rather than any trusted CA.

4. Implementing Certificate Pinning in Android: Provide a technical breakdown of the implementation strategies, covering modern approaches like the Network Security Config (API 24+), OkHttp's CertificatePinner, and custom TrustManager implementations for advanced or legacy use cases.

5. Best Practices, Pitfalls, and Certificate Rotation: Discuss operational realities such as configuring backup pins for seamless certificate rotation, handling pin validation failures gracefully (fail-closed vs. fail-open), and managing debug versus release build configurations to avoid breaking development workflows.

6. Conclusion: Building a Defense-in-Depth Strategy: Summarize how combining HTTPS with certificate pinning creates a robust security posture, emphasizing that pinning is a critical layer in protecting user data on Android.

============================================================

Expanding: 1. The Threat Landscape: Why Network Security Matt...
Expanding: 2. Establishing the Baseline with HTTPS and TLS: E...
Expanding: 3. The Limitations of HTTPS and the Case for Pinni...
Expanding: 4. Implementing Certificate Pinning in Android: Pr...
Expanding: 5. Best Practices, Pitfalls, and Certificate Rotat...
Expanding: 6. Conclusion: Building a Defense-in-Depth Strateg...

============================================================

FINAL ARTICLE:
# Securing Android Network Traffic: HTTPS and Certificate Pinning for Developers

## Introduction

Every Android application that communicates with a backend server is, by default, sending data across a network that no developer controls. Whether that traffic carries authentication tokens, personal health records, or payment details, it traverses Wi-Fi access points, cellular towers, corporate proxies, and intermediate routers—any one of which could be compromised or actively hostile. For the Android developer, the temptation is to treat the network as a reliable pipe and focus on building features. But this assumption is dangerously naive, and the consequences of ignoring it are well-documented: stolen credentials, leaked user data, and reputational damage that can sink an app's reputation overnight.

The good news is that the Android platform provides a rich set of tools to defend against these threats, from built-in TLS enforcement to declarative security configurations and programmatic pinning mechanisms. This article walks through those tools in a progressive, layered fashion. We begin by examining the threat landscape—understanding exactly what attackers can do and why standard HTTPS alone is not always enough. From there, we establish the baseline with HTTPS and TLS, explore the Network Security Configuration file that lets you customize transport-layer policy without writing code, and then dive deep into certificate pinning: what it is, why it matters, and how to implement it using modern Android APIs and popular libraries like OkHttp. Finally, we cover the operational realities—backup pins, certificate rotation, debug-versus-release configurations, and graceful failure handling—that separate a pinning implementation that works from one that breaks your app in production. By the end, you will have a comprehensive understanding of how to build a defense-in-depth network security strategy that protects your users' data from the most common and sophisticated transport-layer attacks.

---

## 1. The Threat Landscape: Why Network Security Matters

Every time an Android app makes a network request—whether it's authenticating a user, fetching sensitive data from an API, or submitting a payment—it sends information across an inherently untrusted medium: the public internet. For developers, it's easy to treat the network as a reliable pipe and focus on building features, but this assumption is dangerously naive. The reality is that mobile devices operate in unpredictable and hostile network environments. An Android phone might connect to a rogue access point at a coffee shop, traverse a corporate proxy that intercepts and inspects TLS traffic, or even share a local network with a malicious actor running a tool like `ettercap` or `mitmproxy`. In each of these scenarios, the attacker can position themselves between the app and its server, silently intercepting, reading, and potentially modifying every byte of unencrypted or improperly secured traffic. This class of attack is known as a Man-in-the-Middle (MitM) attack, and it remains one of the most practical and devastating threats to mobile application security.

The consequences of an unmitigated MitM attack on an Android app are severe and immediate. Consider an app that transmits authentication tokens, personally identifiable information (PII), or financial data over an unprotected HTTP connection. An attacker on the same Wi-Fi network can use packet sniffing tools like Wireshark to capture this traffic in plaintext, extracting session cookies, API keys, or user credentials with minimal effort. Even when HTTPS is in use, a compromised or malicious Certificate Authority (CA)—or a corporate device management policy that installs a custom root CA on the user's device—can enable an attacker to forge valid certificates for any domain, effectively making the MitM invisible to the app's standard TLS validation logic. Android's open ecosystem amplifies this risk: users can sideload apps, install custom CA certificates for debugging or enterprise reasons, and grant apps broad network permissions, all of which expand the attack surface. For the developer, the takeaway is clear: securing data in transit is not an optional enhancement or a compliance checkbox—it is a foundational requirement of any app that handles user data or communicates with backend services.

The Android platform provides powerful built-in mechanisms to defend against these threats, but they are not always enabled or configured correctly by default. Understanding the threat landscape is the essential first step toward making informed decisions about how to protect your app's network traffic. In the sections that follow, we will explore how HTTPS and TLS establish the baseline for encrypted communication, why that baseline alone is insufficient in certain threat models, and how certificate pinning can be used to close the remaining gaps—giving developers the tools to build a truly defense-in-depth network security strategy for their Android applications.

---

## 2. Establishing the Baseline with HTTPS and TLS

At its core, HTTPS is HTTP layered over TLS (Transport Layer Security), which provides three essential guarantees for network communication: encryption, data integrity, and server authentication. When an Android app initiates an HTTPS connection, the TLS handshake negotiates a symmetric session key between the client and server, after which all application-layer data is encrypted before transit and decrypted only upon arrival. The server presents a certificate chain that the Android device validates against its built-in set of trusted Certificate Authorities (CAs)—a root store maintained and updated by Google through Android system updates. This chain-of-trust model ensures that the app is communicating with the legitimate server and that no intermediary has tampered with the data in transit. Since Android 9 (API level 28), the platform enforces HTTPS by default: cleartext (unencrypted HTTP) traffic is blocked unless the developer explicitly opts in, providing a strong baseline that protects user data even if the developer makes no additional configuration changes.

To customize and refine this baseline, Android provides the **Network Security Configuration** file—an XML resource declared in the app's `AndroidManifest.xml` via the `android:networkSecurityConfig` attribute. This file allows developers to define granular security policies without writing any code. Within it, you can specify custom trust anchors (such as pinning specific certificates or CAs), control whether cleartext traffic is permitted per domain or globally using the `<cleartextTrafficPermitted>` flag, and configure debug-only overrides that relax security rules during development without affecting release builds. For example, a developer might allow cleartext traffic to a local emulator or staging server in debug mode while enforcing strict HTTPS in production, all through declarative configuration. The file also supports `<domain-config>` elements that let you apply different security rules to different domains—critical for apps that interact with multiple backend services or third-party APIs with varying security postures.

The Network Security Configuration file is not merely a convenience; it is the recommended and most maintainable way to manage an app's transport-layer security policy. It integrates with Android's build system so that debug and release variants can inherit from a shared base configuration while overriding specific settings as needed. Developers can also reference system-trusted CAs, user-installed CAs, or system-preinstalled certificates, giving fine-grained control over what the app considers trustworthy. By properly configuring this file—explicitly declaring which domains may use cleartext, specifying custom trust stores where necessary, and leveraging the built-in defaults that block insecure traffic—developers establish a solid, auditable foundation upon which additional protections like certificate pinning can be layered in subsequent stages of the security strategy.

---

## 3. The Limitations of HTTPS and the Case for Pinning

While HTTPS encrypts data in transit and authenticates the server through a chain of trust rooted in Certificate Authorities (CAs), this model has inherent weaknesses that are often overlooked. The entire trust hierarchy depends on the assumption that every CA in the system is honest and secure—a fragile premise. History has shown this assumption can fail: the 2011 DigiNotar breach demonstrated that a compromised CA could issue fraudulent certificates for any domain, enabling undetectable MitM attacks against any app relying on standard certificate validation. Similarly, in enterprise environments, corporate proxy solutions like Zscaler or Burp Suite routinely intercept HTTPS traffic by installing a custom root CA certificate on managed devices, effectively allowing the organization to decrypt and inspect all encrypted traffic. In both scenarios, the app's TLS handshake succeeds without any warning to the user or the developer, because the presented certificate chains back to a trusted root—whether that root is legitimately trusted or maliciously injected.

Certificate pinning addresses these vulnerabilities by decoupling the app's trust from the broad CA ecosystem and instead binding it to a specific set of known, expected certificates or their corresponding public keys. Rather than accepting any certificate signed by a trusted CA, the app compares the server's certificate or public key against a pre-configured set of pins during the handshake. If the presented credential does not match one of the pinned values, the connection is rejected outright, regardless of whether the certificate would otherwise be considered valid by the system's trust store. This means that even if an attacker obtains a fraudulent certificate from a compromised CA, or a corporate proxy attempts to intercept traffic with its own CA, the pinning check will fail and the connection will be terminated—preventing the data from ever being exposed.

There are two primary pinning strategies developers should understand: certificate pinning and public key pinning. Certificate pinning directly matches the full X.509 certificate or a subset of its fields, which is straightforward but brittle because certificates expire and must be rotated. Public key pinning, on the other hand, extracts and matches the public key from the certificate, which is more resilient to certificate rotation since the key pair can remain stable across certificate renewals. For Android developers, both approaches can be implemented through the Network Security Configuration file (API 24+) using `<pin-set>` directives, or programmatically via OkHttp's `CertificatePinner` class, which allows specifying SHA-256 hashes of the expected certificates or public keys as base64-encoded pins. The key takeaway is that pinning transforms the security model from "trust any CA" to "trust only these specific credentials," closing a significant gap that standard HTTPS alone leaves open.

---

## 4. Implementing Certificate Pinning in Android

The most straightforward and recommended approach for modern Android applications is leveraging the **Network Security Configuration** file, introduced in API 24 (Android 7.0). This declarative XML-based mechanism allows developers to pin certificates or public keys directly in the app's resources without writing a single line of networking library code. In the `res/xml/network_security_config.xml` file, you define a `<pin-set>` within a `<domain-config>` block, specifying the SHA-256 hashes of your server's certificate and its backup pins. For example, you would include `<pin digest="SHA-256">BASE64_HASH_OF_YOUR_CERT</pin>` entries and set `includeCerts="true"` to enforce that only those specific certificates are trusted for the given domain. The system's `TrustManager` automatically enforces these pins at the TLS handshake layer, meaning every `HttpsURLConnection`, `OkHttp`, `Retrofit`, or `WebView` instance that respects the network security config will automatically validate pinned certificates—no manual interceptor or trust manager wiring is required. This approach also supports specifying pins for debug builds separately, allowing developers to use a different set of pins (or none at all) when running against a staging server, which prevents the common pitfall of pinning production certificates in a development environment.

For applications that rely on **OkHttp** as their HTTP client—which is the de facto standard in the Android ecosystem—OkHttp provides a first-class `CertificatePinner` API that offers fine-grained, programmatic control over pinning logic. You instantiate a `CertificatePinner` by calling `CertificatePinner.Builder()`, chaining `.add()` calls for each domain with their respective SHA-256 (or SHA-1) pin hashes, and then passing the pinner into your `OkHttpClient` builder via `.certificatePinner()`. OkHttp then validates the server's certificate chain against these pins during every handshake, throwing an `SSLPeerUnverifiedException` if no certificate in the chain matches a registered pin. This approach is particularly powerful because it works independently of the Network Security Config, giving you the ability to pin certificates for domains that are dynamically configured or loaded at runtime, and it integrates seamlessly with OkHttp's interceptor chain, allowing you to attach logging or retry logic alongside your pinning enforcement.

For scenarios requiring maximum flexibility—such as legacy apps targeting API levels below 24, apps using custom TLS stacks, or those needing to implement certificate pinning across multiple HTTP clients uniformly—a **custom `TrustManager`** implementation provides the lowest-level control. By creating a `TrustManagerFactory` that wraps the default `X509TrustManager` and overrides the `checkServerTrusted()` method to perform manual pin validation against the server's certificate chain, you can enforce pinning regardless of the HTTP library in use. In this approach, you extract the leaf certificate from the `X509Certificate[]` chain passed to `checkServerTrusted()`, compute its SHA-256 hash, and compare it against a hardcoded or remotely fetched set of known-good pins, throwing a `CertificateException` if no match is found. While this method demands careful implementation to avoid introducing subtle security flaws—such as accidentally falling back to the default trust store without pin validation—it is indispensable for apps that need to pin certificates in contexts where declarative configuration or library-level APIs are insufficient, such as WebSocket connections, gRPC channels, or custom TLS sockets used in native code via the NDK.

---

## 5. Best Practices, Pitfalls, and Certificate Rotation

One of the most operationally critical aspects of certificate pinning is planning for certificate rotation long before it becomes an emergency. Certificates expire, and CAs may rotate their root or intermediate keys on schedules you don't control—so if your app pins exclusively to a single certificate or public key, you risk a hard failure the moment that key changes. The solution is to configure backup pins alongside your primary pin, which in Android's Network Security Config is as simple as declaring multiple `<pin>` elements within a `<certificates>` block. The system will accept a connection if any one of the declared pins matches the server's certificate chain. When it's time to rotate, you ship an update that adds the new certificate's pin as a backup (or primary) while keeping the old pin temporarily in the list, then follow up with a subsequent update that removes the deprecated pin. This two-step rollout window—often spanning weeks or months—ensures that users on older app versions don't experience a connectivity blackout. For teams using OkHttp's `CertificatePinner`, the same principle applies: you can call `CertificatePinner.Builder().add(hostname, pin1).add(hostname, pin2)` to register multiple pins per host, and you should instrument your monitoring to alert when a pin mismatch occurs so you can trigger the rollout process before a certificate expiry catches you off guard.

Equally important is deciding how your app should behave when pin validation fails—a decision that carries significant security tradeoffs. A **fail-closed** approach means that any pin mismatch results in an immediate and hard rejection of the connection, refusing to establish a secure channel under any circumstances. This is the correct default for production builds handling sensitive data, as it guarantees that a compromised or misconfigured certificate chain cannot silently slip through. However, fail-closed behavior can be devastating during development or in edge-case scenarios like a server serving a dynamically generated certificate from a local test CA. This is where the distinction between debug and release build configurations becomes essential. In your `res/xml/network_security_config.xml`, you can use the `<debug-overrides>` element to specify a less restrictive trust configuration that applies only when the app is built in debug mode—allowing developers to work with self-signed certs or corporate proxies without triggering pin failures. In release builds, you should enforce strict pinning with fail-closed behavior and ensure that the debug overrides are completely stripped from the production APK. For OkHttp-based implementations, a common pattern is to build two separate `OkHttpClient` instances: one with a permissive `TrustManager` and no `CertificatePinner` for debug, and another with a custom `TrustManager` that enforces your pin set for release.

A subtle but frequently overlooked pitfall is the risk of pinning to a certificate rather than a public key digest. Pinning to a certificate (using the `android:pin` attribute or OkHttp's base64-encoded SHA-256 of the certificate) ties your app to the entire certificate including its validity period and serial number, meaning any re-issuance—even with the same public key—will break the pin. Pinning to a public key digest (`android:pinDigest="SHA-256"` in Network Security Config, or the `sha256/...` format in OkHttp) is more resilient because it focuses on the cryptographic identity that actually matters, surviving certificate re-issuance as long as the public key remains unchanged. This distinction can save your team from an emergency patch every time a certificate is reissued due to routine CA operations. Additionally, you should always log pin failure events with enough context—hostname, presented certificate fingerprint, expected pin—to diagnose issues in production without exposing sensitive data, and consider implementing a remote configuration flag that can temporarily relax pinning for a subset of users if an unforeseen rotation incident occurs, giving you a controlled escape valve rather than a blanket outage.

---

## 6. Conclusion: Building a Defense-in-Depth Strategy

In practice, securing Android network traffic is not a single decision but a layered strategy. HTTPS and TLS provide the essential foundation by encrypting data in transit and authenticating servers against a chain of trusted Certificate Authorities. However, as we have seen, this baseline alone leaves an app exposed to scenarios where a CA is compromised, a corporate proxy intercepts traffic, or an attacker installs a rogue root certificate on a device. Certificate pinning closes this gap by ensuring that the app trusts only the specific cryptographic identities it expects, effectively reducing the attack surface from thousands of trusted CAs to a handful of known, pinned certificates or public keys. Together, these two mechanisms form a defense-in-depth approach: HTTPS protects against passive eavesdropping and basic impersonation, while pinning adds a critical safeguard against sophisticated trust-chain attacks that would otherwise go undetected.

For developers building production Android applications, the takeaway is clear—relying solely on the default trust store is no longer sufficient for apps handling sensitive user data. By leveraging the Network Security Configuration file to enforce HTTPS and define pin sets, integrating OkHttp's `CertificatePinner` for fine-grained control, and implementing graceful failure handling with backup pins for certificate rotation, teams can build a network security posture that is both resilient and maintainable. It is equally important to manage the operational lifecycle of pins carefully, ensuring that debug builds do not inadvertently enforce production-level pinning and that certificate rotations are planned well in advance with overlapping backup pins to avoid service disruptions.

Ultimately, certificate pinning is not a silver bullet, but it is a powerful and necessary layer in a comprehensive security strategy. When combined with HTTPS, it significantly raises the bar for an attacker attempting to intercept or manipulate an app's network communications. In a landscape where mobile devices routinely carry personal, financial, and health-related data, the cost of neglecting these protections far outweighs the engineering effort required to implement them. By treating network security as a first-class concern—from initial architecture through ongoing maintenance—Android developers can deliver applications that users can trust with their most sensitive information.



"""