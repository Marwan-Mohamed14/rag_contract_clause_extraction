"""
=============================================================
  Legal Basis Generator — Rule-Based Only
  Version: 2.0

  - 100% rule-based, no API, no cost
  - Format: "Article X — what the article says; how it applies"
  - Only processes entries missing legal_basis
  - Saves progress after every 50 entries (crash-safe)
=============================================================
"""

import json
import os

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

INPUT_FILE    = r"C:\Users\Mohamed\Desktop\better_call_maro\generated_clauses\caluses.json"
OUTPUT_FILE   = r"C:\Users\Mohamed\Desktop\better_call_maro\generated_clauses\clauses_with_legal_basis.json"
PROGRESS_FILE = r"C:\Users\Mohamed\Desktop\better_call_maro\generated_clauses\progress_backup.json"

SAVE_EVERY = 50   # save progress every N entries

# ─────────────────────────────────────────────
#  APPROACH 1 — RULE-BASED LEGAL BASIS MAP
#  Format: violation_type → legal_basis string
#  Covers all violation types we confirmed in previous sessions
# ─────────────────────────────────────────────

LEGAL_BASIS_MAP = {

    # ── Label 0 (compliant) ──────────────────────────────────────────────────
    "none": "",  # compliant entries get legal_basis from clause context — handled separately

    # ── Contract formation / effectivity ────────────────────────────────────
    "unilateral_contract_formation": (
        "Article 89 — contract requires mutual exchange of consent from both parties; "
        "one party cannot unilaterally declare the contract effective or bind the other party without their expressed consent"
    ),
    "retroactive_contract_formation": (
        "Article 89 — contract concluded upon mutual exchange of consent at the time of signing; "
        "imposing retroactive effect before consent was exchanged contradicts this requirement"
    ),
    "incomplete_preliminary_agreement": (
        "Article 101 — a preliminary agreement to conclude a future contract is not binding "
        "unless all essential terms and the time period are specified; "
        "deferring essential terms renders the preliminary agreement unenforceable"
    ),
    "implied_consent_substituting_written_consent": (
        "Article 89 — mutual consent required; "
        "Article 90 — parties may stipulate that consent must be express rather than implied; "
        "silence, inaction, or conduct does not substitute for written mutual consent"
    ),
    "missing_mutual_consent": (
        "Article 89 — contract requires mutual exchange of consent expressed by both parties; "
        "one party's signature or action alone cannot bind the other party without their consent"
    ),
    "missing_written_formality": (
        "Article 89 — mutual consent required expressed in writing; "
        "Article 90 — written expression of intent is the valid form of consent; "
        "verbal, email, or click-based confirmation does not satisfy the written formality requirement"
    ),
    "missing_essential_terms": (
        "Article 101 — preliminary agreement must specify all essential terms including price, scope, "
        "and timeline to be binding; "
        "Article 150 — contract terms must be sufficiently defined to be enforced"
    ),
    "unilateral_formality_control": (
        "Article 89 — mutual consent required; "
        "Article 150 — clear contract terms shall not be deviated from; "
        "allowing one party to control essential formalities unilaterally undermines mutual consent"
    ),
    "implied_consent_without_express_agreement": (
        "Article 89 — mutual consent required; "
        "Article 90 — implied consent requires conduct leaving no doubt as to meaning; "
        "conduct, use of service, or payment does not substitute for express written consent"
    ),
    "unilateral_withdrawal_without_compensation": (
        "Article 89 — contract obligations arise upon mutual consent; "
        "Article 157 — bilateral contracts require notice before dissolution; "
        "allowing unilateral withdrawal without compensation contradicts these protections"
    ),
    "unilateral_modification_of_essential_terms": (
        "Article 150 — clear contract terms shall not be deviated from under pretext of interpretation; "
        "one party cannot unilaterally modify essential terms such as price or scope after signing"
    ),
    "unilateral_contract_interpretation": (
        "Article 150 — common intention of parties shall be sought in interpretation; "
        "clear terms shall not be deviated from; "
        "one party cannot unilaterally determine the interpretation of disputed terms"
    ),
    "deviation_from_clear_terms": (
        "Article 150 — where contract terms are clear and unambiguous they shall not be deviated from "
        "under the pretext of interpretation by either party"
    ),
    "excluding_judicial_discretion": (
        "Article 157 — court retains discretion to refuse dissolution for minor breach; "
        "any agreement to exclude court discretion for minor breaches is contrary to this statutory protection"
    ),
    "unlawful_earnest_money_restriction": (
        "Article 103 — earnest money at contract conclusion entitles either party to withdraw; "
        "restricting the right to withdraw from an earnest money arrangement contradicts this statutory right"
    ),
    "waiver_of_enforcement_right": (
        "Article 102 — if one party refuses to conclude the contract the other may seek court enforcement; "
        "waiving the right to seek court enforcement of a preliminary agreement is void"
    ),
    "dissolution_without_notice": (
        "Article 157 — bilateral contracts require prior written notice to the defaulting party "
        "before dissolution takes effect; automatic dissolution without notice violates this requirement; "
        "Article 158 — notice waiver only permitted if expressly stated in contract"
    ),
    "dissolution_without_notice_or_warning": (
        "Article 157 — prior notice required before dissolution; "
        "Article 158 — notice waiver must be expressly stated; "
        "immediate rescission without warning or opportunity to cure contradicts both articles"
    ),

    # ── Payment / delivery ───────────────────────────────────────────────────
    "payment_before_delivery": (
        "Article 656 — remuneration is payable upon delivery unless otherwise agreed; "
        "requiring payment before the work is made available to the employer contradicts this rule"
    ),
    "payment_before_formal_delivery_acceptance": (
        "Article 656 — remuneration payable upon delivery; "
        "Article 655 — delivery requires formal acceptance by employer; "
        "triggering payment before formal delivery acceptance circumvents both articles"
    ),
    "payment_regardless_of_completion": (
        "Article 656 — remuneration payable upon delivery of completed work; "
        "requiring payment regardless of whether the work has been completed and made available "
        "contradicts the delivery-linked payment rule"
    ),
    "unilateral_payment_schedule": (
        "Article 690 — payment must occur at the time and place specified in the contract or by custom; "
        "allowing one party to unilaterally determine payment time or place after signing "
        "overrides this statutory requirement"
    ),
    "payment_acceleration_without_notice": (
        "Article 218 — compensation not due before notice unless debtor expressly refuses in writing; "
        "Article 656 — remuneration payable upon delivery; "
        "accelerating full payment without notice upon unjustified refusal violates both"
    ),
    "payment_on_contractor_delay": (
        "Article 656 — remuneration payable upon delivery; "
        "requiring full payment when contractor is responsible for the delay contradicts the "
        "delivery-linked payment principle"
    ),
    "conditional_delivery_on_prepayment": (
        "Article 655 — contractor must make work available to employer; "
        "Article 656 — payment due upon delivery not before; "
        "conditioning delivery on prepayment inverts the statutory sequence"
    ),
    "removal_of_employer_delivery_rights": (
        "Article 655 — employer has the right to inspect and take formal delivery; "
        "removing the employer's right to refuse delivery under any circumstance "
        "eliminates the statutory delivery inspection right"
    ),
    "waiver_of_delivery_linked_payment_right": (
        "Article 656 — remuneration linked to delivery; "
        "waiving the employer's right to withhold payment pending formal delivery "
        "eliminates the statutory delivery-payment linkage"
    ),
    "deemed_delivery_on_justified_refusal": (
        "Article 655 — work deemed delivered only upon unjustified refusal after formal notification; "
        "treating any refusal including justified refusal for defects as unjustified "
        "removes the employer's legitimate right to refuse defective work"
    ),
    "deemed_delivery_without_reasonable_notice_period": (
        "Article 655 — formal notification required before deemed delivery applies; "
        "a notice period of less than a reasonable time (12-24 hours) does not satisfy "
        "the formal notification requirement"
    ),
    "payment_demanded_without_delivery": (
        "Article 656 — remuneration payable upon delivery; "
        "demanding full remuneration while refusing or withholding delivery "
        "contradicts the delivery-linked payment rule"
    ),
    "waiver_of_fair_compensation_on_contractor_incapacity": (
        "Article 656 — remuneration tied to completion and delivery; "
        "where contractor becomes incapable of completing work, employer is entitled to "
        "fair compensation and return of materials; waiving this right is contrary to Egyptian law"
    ),

    # ── Delivery acceptance / defects ────────────────────────────────────────
    "waiver_of_rectification_warning_right": (
        "Article 650 — employer must warn contractor of defective performance before seeking rescission; "
        "waiving the right to issue such a warning eliminates the employer's statutory remedy sequence"
    ),
    "waiver_of_rescission_right": (
        "Article 650 — employer may seek rescission if contractor fails to rectify defective performance; "
        "waiver of the rescission right eliminates this mandatory statutory remedy"
    ),
    "waiver_of_alternative_completion_right": (
        "Article 650 — employer may engage substitute contractor at first contractor's expense "
        "if rectification is refused or impossible; waiving this right eliminates the "
        "employer's statutory remedy for non-performance"
    ),
    "removal_of_employer_delivery_inspection_right": (
        "Article 655 — employer has the right to inspect before formal acceptance; "
        "requiring immediate acceptance upon notification without inspection removes this right"
    ),
    "removal_of_reasonable_rectification_period": (
        "Article 650 — employer sets a reasonable period for rectification; "
        "requiring an unreasonably long rectification period or allowing contractor to set "
        "unlimited rectification time removes the employer's right to set a reasonable period"
    ),
    "waiver_of_defect_liability_period": (
        "Article 651 — architect and contractor jointly liable for structural defects for ten years; "
        "limiting or waiving post-delivery defect liability contradicts this mandatory provision"
    ),
    "removal_of_defect_claim_right": (
        "Article 650 — employer retains right to claim for defects; "
        "Article 651 — ten-year liability for structural defects; "
        "barring defect claims after unreasonably short periods contradicts both articles"
    ),
    "contractor_consent_required_for_rescission": (
        "Article 650 — employer may seek rescission without requiring contractor consent; "
        "conditioning rescission on contractor's written consent eliminates this unilateral statutory remedy"
    ),
    "unilateral_delivery_declaration": (
        "Article 655 — delivery requires formal notification to employer and opportunity to accept; "
        "contractor cannot unilaterally declare work delivered without making it physically available "
        "or providing formal written notification"
    ),
    "removal_of_formal_notification_requirement": (
        "Article 655 — formal written notification required before delivery obligations arise; "
        "replacing written notification requirement with verbal notification removes the "
        "statutory formality requirement"
    ),
    "removal_of_immediate_rescission_right_when_rectification_impossible": (
        "Article 650 — where rectification is clearly impossible employer may seek immediate rescission; "
        "requiring an extended rectification period even when rectification is impossible "
        "contradicts this statutory exception"
    ),
    "removal_of_reasonable_acceptance_period": (
        "Article 655 — employer entitled to a reasonable period to inspect and accept delivery; "
        "requiring acceptance within an unreasonably short period (less than reasonable time) "
        "contradicts the employer's inspection right"
    ),

    # ── Environment preparation ──────────────────────────────────────────────
    "unilateral_modification_of_approved_requirements": (
        "Article 150 — clear contract terms including approved requirements shall not be deviated from; "
        "allowing one party to modify approved requirements after the other party has begun "
        "preparation contradicts this rule and the mutual consent requirement"
    ),
    "refusal_of_delivery_while_demanding_payment": (
        "Article 656 — remuneration payable upon delivery; "
        "refusing delivery while simultaneously demanding full payment directly contradicts "
        "the delivery-linked payment rule"
    ),
    "waiver_of_rectification_period": (
        "Article 650 — employer must set a reasonable period for contractor to rectify defects; "
        "waiving the right to receive a reasonable rectification period removes the employer's "
        "statutory remedy sequence"
    ),
    "alternative_completion_without_prior_warning": (
        "Article 650 — employer must warn contractor before engaging substitute for completion; "
        "engaging a third party without prior warning to the contractor skips the mandatory "
        "warning step in the statutory remedy sequence"
    ),
    "unilateral_acceptance_determination": (
        "Article 655 — acceptance determined objectively upon completion and formal notification; "
        "allowing one party to unilaterally determine whether delivery meets requirements "
        "without objective criteria contradicts the statutory delivery standard"
    ),
    "contractor_delay_without_liability": (
        "Article 656 — remuneration tied to delivery; "
        "Article 650 — contractor liable for defective or late performance; "
        "allowing contractor to delay delivery indefinitely without liability for the delay "
        "contradicts both articles"
    ),
    "alternative_completion_without_agreed_compensation_terms": (
        "Article 650 — substitute completion at first contractor's expense; "
        "allowing substitute completion at rates determined solely by the substituting party "
        "without prior agreement contradicts the requirement for agreed compensation terms"
    ),
    "refusal_without_formal_notification": (
        "Article 655 — formal written notification required for both delivery and refusal; "
        "refusing delivery without formally notifying the other party of specific deficiencies "
        "contradicts the notification requirement"
    ),
    "removal_of_deemed_delivery_on_unjustified_refusal": (
        "Article 655 — work deemed delivered upon unjustified refusal after formal notification; "
        "removing the deemed delivery protection when contractor unjustifiably refuses "
        "eliminates this statutory protection for the employer"
    ),
    "waiver_of_compensation_for_unjustified_contractor_delay": (
        "Article 650 — employer entitled to compensation for contractor's defective or late performance; "
        "waiving compensation for delays caused by unjustified contractor refusal "
        "eliminates this statutory entitlement"
    ),
    "immediate_rescission_when_rectification_possible": (
        "Article 650 — employer must first warn contractor and allow rectification before rescission; "
        "allowing immediate rescission when rectification is possible skips the mandatory "
        "warning and rectification step"
    ),
    "unilateral_retraction_of_formal_acceptance": (
        "Article 655 — formal acceptance once given is binding unless obtained by fraud or error; "
        "allowing one party to retract formal acceptance unilaterally after delivery "
        "contradicts the binding nature of formal acceptance"
    ),

    # ── IP and data ──────────────────────────────────────────────────────────
    "unauthorized_data_use_without_consent": (
        "Article 2 — personal data may only be processed with explicit consent of the data subject; "
        "Article 36 — unauthorized processing carries criminal penalty of EGP 100,000–1,000,000"
    ),
    "data_without_consent": (
        "Article 2 — personal data requires explicit consent of data subject specifying purpose; "
        "collecting or processing data without authorization violates this requirement"
    ),
    "data_monetization": (
        "Article 2 — data may only be used for the purpose consented to; "
        "Article 36 — unauthorized commercial use of data carries criminal penalty; "
        "selling or commercializing personal data without explicit consent is prohibited"
    ),
    "cross_border_data": (
        "Article 26 — personal data may not be transferred outside Egypt except under specific conditions; "
        "transferring or storing data on servers outside Egypt without authorization violates "
        "Egypt's Personal Data Protection Law No. 151 of 2020"
    ),
    "unauthorized_ip_ownership_transfer": (
        "Article 149 — economic rights of author cannot be transferred without explicit written agreement; "
        "Article 150 — fair remuneration required for transfer of economic rights; "
        "automatic or implied transfer of IP ownership contradicts both articles"
    ),
    "ip_transfer": (
        "Article 149 — author retains all economic rights not explicitly signed away in writing; "
        "Article 150 — transfer of economic rights requires explicit written authorization and fair remuneration; "
        "automatic transfer of intellectual property upon creation without compensation is void"
    ),
    "waiver_of_remuneration_right": (
        "Article 150 — author entitled to remuneration he considers fair for transfer of economic rights; "
        "waiver of this entitlement as a condition of engagement is void; "
        "Article 36 — unauthorized use of economic rights carries criminal penalty"
    ),
    "implied_consent_for_ip_transfer": (
        "Article 149 — economic rights must be explicitly signed away; submission does not constitute transfer; "
        "Article 2 — explicit consent required; participation or submission is not implied consent"
    ),
    "unauthorized_use_of_ideas_without_consent_or_remuneration": (
        "Article 149 — author retains economic rights not explicitly transferred; "
        "Article 150 — remuneration required for use of economic rights; "
        "using submitted ideas or materials without consent or remuneration violates both"
    ),

    # ── Representative liability ─────────────────────────────────────────────
    "exemption_from_representative_liability": (
        "Article 662 — contractor remains liable towards employer for work of subcontractors; "
        "exempting a party from liability for subcontractor or representative breaches "
        "directly contradicts this mandatory principle"
    ),
    "conditional_exemption_from_representative_liability": (
        "Article 662 — contractor's liability for subcontractor work is unconditional; "
        "conditioning exemption from representative liability on specific circumstances "
        "contradicts the primary and unconditional nature of this liability"
    ),
    "secondary_rather_than_primary_representative_liability": (
        "Article 662 — contractor's liability to employer for subcontractor work is primary and direct; "
        "making representative liability secondary or conditional on employer first pursuing "
        "the representative directly contradicts this primary liability rule"
    ),
    "representative_liability_cap_without_gross_negligence_override": (
        "Article 662 — contractor remains liable for subcontractor work; "
        "capping representative liability and waiving excess claims limits the statutory "
        "primary liability principle"
    ),
    "time_limited_exemption_from_representative_liability": (
        "Article 662 — contractor remains liable for subcontractor work without time limitation; "
        "introducing time-based diminution of representative liability contradicts this principle"
    ),
    "representative_liability_scope_limitation": (
        "Article 662 — contractor remains fully liable for subcontractor work; "
        "restricting representative liability to direct losses only and excluding indirect losses "
        "limits the scope of the mandatory primary liability"
    ),

    # ── Severability / interpretation ────────────────────────────────────────
    "invalidity_of_part_voids_whole_agreement": (
        "Article 148 — contract obligations include what is implied by law and equity; "
        "Article 150 — remaining clear terms shall not be deviated from; "
        "voiding the entire agreement upon partial invalidity contradicts the good faith "
        "preservation principle"
    ),
    "unilateral_interpretation_in_drafter_favor": (
        "Article 150 — common intention of both parties shall be sought in interpretation; "
        "Article 151 — ambiguity shall be interpreted in favor of the debtor not the drafter; "
        "interpreting remaining provisions in the drafter's favor contradicts both principles"
    ),
    "exclusion_of_intent_based_interpretation": (
        "Article 150 — where room for interpretation exists common intention of parties shall be sought; "
        "excluding original intent from interpretation directly contradicts this mandatory principle"
    ),
    "unilateral_termination_on_partial_invalidity": (
        "Article 148 — good faith performance of remaining obligations required; "
        "unilateral termination right on partial invalidity with no liability contradicts "
        "the good faith obligation"
    ),
    "unilateral_determination_of_severability": (
        "Article 150 — common intention of both parties guides interpretation and replacement; "
        "Article 148 — good faith requires mutual consideration; "
        "allowing one party to unilaterally determine how remaining provisions apply "
        "contradicts both principles"
    ),
    "exclusion_of_implied_obligations_after_partial_invalidity": (
        "Article 148 — contract includes obligations implied by law, custom, and equity; "
        "excluding implied obligations after partial invalidity directly contradicts "
        "this mandatory principle"
    ),
    "exclusion_of_good_faith_obligation": (
        "Article 148 — contract must be performed in good faith; this is a mandatory principle "
        "that cannot be contracted out of; any agreement to exclude good faith obligation is void"
    ),
    "waiver_of_modification_right_after_partial_invalidity": (
        "Article 149 — any agreement to the contrary to court's right to modify unfair terms is void; "
        "Article 151 — in adhesion contracts ambiguous terms may not be interpreted to detriment "
        "of adhering party; waiving modification rights is void under both"
    ),

    # ── Document hierarchy ───────────────────────────────────────────────────
    "inverted_document_hierarchy": (
        "Article 150 — clear contract terms shall not be deviated from; "
        "the Contract body and its formal annexes must prevail over all incorporated documents "
        "such as financial offers and technical proposals"
    ),
    "impermissible_interpretive_deviation": (
        "Article 150 — where contract language is clear and unambiguous it shall be applied as written "
        "without deviation under the pretext of interpretation"
    ),

    # ── Commercial agency / termination ─────────────────────────────────────
    "waiver_of_statutory_termination_right": (
        "Article 163 — either party to a commercial agency contract may terminate at any time; "
        "irrevocable waiver of the statutory termination right is void"
    ),
    "exclusion_of_statutory_termination_compensation": (
        "Article 163 — compensation is mandatory where termination occurs without prior notice "
        "or at a commercially inconvenient time; waiver of this compensation obligation is void"
    ),
    "restriction_of_statutory_termination_compensation": (
        "Article 163 — compensation for termination without notice is a statutory right; "
        "restricting or capping this compensation below the statutory standard is void"
    ),
    "unfair_termination": (
        "Article 157 — bilateral contracts require notice before dissolution; "
        "Article 163 — compensation mandatory if termination is without notice or at inconvenient time; "
        "allowing unilateral termination without notice or cause violates both articles"
    ),

    # ── Technology transfer ──────────────────────────────────────────────────
    "foreign_jurisdiction": (
        "Article 87 — Egyptian courts have mandatory jurisdiction over technology transfer disputes; "
        "Egyptian law governs on the merits; any agreement to the contrary is null and void"
    ),
    "foreign_law_substitution": (
        "Article 87 — Egyptian law governs technology transfer contracts on the merits; "
        "substituting a foreign law for Egyptian law as the governing law is null and void"
    ),
    "restriction_on_importer_freedom": (
        "Article 75 — restrictions on the importer's freedom to use, develop, define, "
        "or advertise the production may be annulled by Egyptian courts"
    ),
    "waiver_of_statutory_renegotiation_right": (
        "Article 86 — either party may request termination or renegotiation after five years "
        "if economic conditions require; this right recurs every five years and cannot be waived"
    ),
    "shortened_annulment_period": (
        "Article 54 — annulment of arbitral awards must be sought within ninety days of notification; "
        "contractually shortening this period deprives parties of their statutory rights"
    ),
    "waiver_of_annulment_rights": (
        "Article 54 — pre-award waiver of annulment rights is inadmissible; "
        "any agreement waiving the right to annul an arbitral award before it is issued is void"
    ),
    "failure_to_be_in_writing": (
        "Article 74 — technology transfer contracts must be in writing; "
        "oral or informal agreements do not satisfy the statutory writing requirement and are void"
    ),
    "denial_of_judicial_and_arbitral_access": (
        "Article 87 — Egyptian courts have mandatory jurisdiction; "
        "any clause denying parties access to Egyptian judicial or arbitral process is void"
    ),
    "exclusion_of_egyptian_court_oversight": (
        "Article 87 — Egyptian court oversight is mandatory for technology transfer contracts; "
        "any agreement to exclude Egyptian court oversight is null and void"
    ),

    # ── Warranty / compensation ──────────────────────────────────────────────
    "liability_exemption_for_fraud_and_gross_negligence": (
        "Article 217 — any clause exempting a party from liability for its own fraud or gross negligence "
        "or that of persons it employs is void; this applies to both direct and vicarious liability"
    ),
    "liability_exemption_for_unlawful_acts": (
        "Article 217 — exemption from liability for unlawful acts is absolutely void and cannot be "
        "cured by agreement"
    ),
    "exclusion_of_natural_consequence_damages": (
        "Article 221 — compensation must include all losses and missed gains that are a natural "
        "and foreseeable consequence of non-performance; exclusion of natural consequence damages is void"
    ),
    "exclusion_of_moral_damages": (
        "Article 222 — compensation includes moral damage; "
        "a blanket waiver of moral damage rights is contrary to this mandatory provision"
    ),
    "compensation_without_notice_requirement": (
        "Article 218 — compensation is not owed before the debtor has been formally served with notice; "
        "removing the notice requirement as a precondition for compensation violates this rule"
    ),
    "unfair_liability": (
        "Article 217 — liability exemptions for fraud and gross negligence are void; "
        "creating unfair one-sided liability allocation where one party bears all liability "
        "and the other bears none contradicts Egyptian law's mandatory liability principles"
    ),

    # ── Missing protection / rights waiver ──────────────────────────────────
    "rights_waiver": (
        "Article 150 — contract terms that waive statutory rights are subject to scrutiny; "
        "Article 157 — parties cannot waive rights to notice, compensation, or judicial remedies "
        "without express written agreement; a general or implied rights waiver is insufficient"
    ),
    "missing_protection": (
        "Article 148 — contract must be performed in good faith including implied protections; "
        "omitting protection clauses that are required by Egyptian law does not relieve parties "
        "of their statutory obligations"
    ),
    "invalid_signature": (
        "Article 89 — contract requires mutual exchange of consent expressed in writing; "
        "Article 90 — written expression of intent is the valid form of consent; "
        "typed names, scanned signatures, email confirmations, or click-based acceptance "
        "do not satisfy the formal signature requirement under Egyptian law"
    ),

    # ── Tax obligations ──────────────────────────────────────────────────────
    "tax_obligation_waiver": (
        "Article 690 — payment must comply with Egyptian special laws including tax laws; "
        "all applicable Egyptian taxes must be paid in accordance with Egyptian tax law; "
        "contractual elimination of tax obligations is void"
    ),
    "tax_liability_reversal": (
        "Article 690 — tax payment obligations allocated by Egyptian law cannot be reversed by contract; "
        "the employer bears the statutory tax payment obligation and this cannot be contractually transferred"
    ),
    "foreign_law_governing_tax": (
        "Article 690 — payment obligations including taxes must comply with Egyptian law; "
        "applying foreign law to tax obligations under an Egyptian-governed contract is void"
    ),
    "unilateral_modification": (
        "Article 150 — clear contract terms shall not be deviated from under pretext of interpretation; "
        "one party cannot unilaterally modify contract terms including price, timeline, or scope "
        "after the contract has been agreed"
    ),

    # ── Construction liability ───────────────────────────────────────────────
    "exclusion_of_ten_year_structural_liability": (
        "Article 651 — architect and contractor jointly liable for ten years for structural collapse "
        "and defects in fixed installations; this period cannot be reduced or excluded by contract"
    ),
    "exclusion_of_architect_structural_liability": (
        "Article 651 — architect is jointly liable with contractor for structural defects for ten years; "
        "exempting the architect from this liability is void"
    ),
    "exclusion_of_contractor_structural_liability": (
        "Article 651 — contractor is jointly liable for structural defects for ten years; "
        "exempting the contractor from this liability is void"
    ),
    "waiver_of_compensation_on_non_delivery": (
        "Article 656 — remuneration is payable upon delivery; "
        "where contractor fails to deliver, employer is entitled to compensation; "
        "waiving the right to claim compensation upon non-delivery contradicts this statutory entitlement"
    ),
    "no_compensation_for_incurred_costs": (
        "Article 650 — employer entitled to compensation for losses caused by contractor's failure; "
        "Article 221 — compensation includes actual loss and missed gain as natural consequences of non-performance; "
        "excluding compensation for costs already incurred by the employer contradicts both articles"
    ),
    "waiver_of_dispute_right": (
        "Article 150 — clear contract terms shall not be deviated from; "
        "Article 149 — court may modify unfair terms in accordance with requirements of justice; "
        "waiving the right to dispute contractor's determinations eliminates the employer's "
        "statutory right to judicial review of contractual disputes"
    ),
    "waiver_of_right_to_substitute_completion": (
        "Article 650 — employer may engage a substitute contractor at the first contractor's expense "
        "following failure to rectify; waiving this right eliminates the statutory remedy"
    ),
    "waiver_of_statutory_renegotiation_and_termination_right": (
        "Article 86 — either party may request renegotiation or termination after five years; "
        "combined waiver of both renegotiation and termination rights is void under this article"
    ),

    # ── Elected domicile ─────────────────────────────────────────────────────
    "delayed_domicile_change_notification": (
        "Article 164 — elected domicile must be kept current; "
        "unreasonably long notification periods for domicile changes (standard is 48 hours) "
        "impede valid service of legal documents"
    ),
    "prohibition_on_domicile_change": (
        "Article 164 — parties must be permitted to update their elected domicile; "
        "prohibiting address changes undermines valid service of legal documents"
    ),
    "unreasonable_domicile_change_restriction": (
        "Article 164 — elected domicile must be kept current with prompt written notice; "
        "unreasonable restrictions on domicile changes undermine valid service requirements"
    ),
}

# ─────────────────────────────────────────────
#  COMPLIANT (LABEL 0) LEGAL BASIS BY CATEGORY
#  Used when label=0 and no violation_type
# ─────────────────────────────────────────────

def derive_compliant_legal_basis(text: str) -> str:
    """Derives legal_basis for compliant clauses based on clause content."""
    text_lower = text.lower()

    if any(w in text_lower for w in ["sign", "signature", "execute", "execution", "binding"]):
        return "Article 89 — contract concluded upon mutual exchange of consent; Article 90 — written expression of intent is valid form of consent"

    if any(w in text_lower for w in ["obligation", "no obligation", "purchase order", "formal written contract"]):
        return "No retrieved law prohibits contract formation preconditions; Article 89 — no obligations arise before mutual formal execution"

    if any(w in text_lower for w in ["payment", "egp", "down payment", "installment", "remuneration"]):
        return "Article 656 — remuneration payable upon delivery unless otherwise agreed; Article 690 — payment at time and place specified in contract per Egyptian law"

    if any(w in text_lower for w in ["tax", "taxes", "duties", "egyptian law", "arab republic"]):
        return "Article 690 — payment obligations must comply with Egyptian special laws including tax laws"

    if any(w in text_lower for w in ["implement", "timeline", "days from", "delivery period", "completion"]):
        return "Article 655 — contractor must complete and make work available within agreed timeline; Article 656 — remuneration linked to delivery"

    if any(w in text_lower for w in ["environment", "infrastructure", "deploy", "requirements approved"]):
        return "Article 655 — employer must prepare suitable environment for program per approved requirements before delivery deadline"

    if any(w in text_lower for w in ["warranty", "guarantee", "bug", "repair", "defect"]):
        return "Article 650 — employer may warn contractor and seek rectification; Article 651 — statutory liability for structural defects"

    if any(w in text_lower for w in ["representative", "subcontractor", "agent", "liable for"]):
        return "Article 662 — contractor remains liable towards employer for subcontractor work"

    if any(w in text_lower for w in ["invalid", "unenforceable", "severab", "void"]):
        return "Article 148 — contract must be performed in good faith including remaining valid obligations; Article 150 — common intention of parties guides interpretation"

    if any(w in text_lower for w in ["consent", "explicit", "data", "personal", "information"]):
        return "Article 2 — explicit consent required for data processing; Article 149 — economic rights require explicit written authorization"

    if any(w in text_lower for w in ["intellectual property", "ip", "copyright", "author", "remuneration"]):
        return "Article 149 — author retains all economic rights not explicitly signed away; Article 150 — fair remuneration required for transfer"

    if any(w in text_lower for w in ["termination", "terminate", "compensation", "notice"]):
        return "Article 157 — bilateral contracts require notice before dissolution; Article 163 — compensation mandatory where termination is without notice"

    if any(w in text_lower for w in ["domicile", "address", "notification", "notify"]):
        return "Article 164 — elected domicile must be kept current; prompt written notice required for address changes"

    # Generic fallback
    return "Article 148 — contract must be performed in good faith; Article 150 — common intention of parties governs interpretation"


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Legal Basis Generator — Rule-Based Only")
    print("="*60)

    # Load input file
    print(f"\n[1] Loading: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"  Total entries: {len(entries)}")

    # Check for existing progress
    if os.path.exists(PROGRESS_FILE):
        print(f"\n  Found progress backup: {PROGRESS_FILE}")
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            processed = json.load(f)
        print(f"  Resuming from entry {len(processed)}")
    else:
        processed = []

    # Count entries needing legal_basis
    needs_basis = [e for e in entries if not e.get("legal_basis", "").strip()]
    print(f"  Entries missing legal_basis: {len(needs_basis)}")
    print(f"  Entries already have legal_basis: {len(entries) - len(needs_basis)}")

    # Stats
    rule_based_count = 0
    truly_unknown = []

    # Process each entry
    start_idx = len(processed)
    for i, entry in enumerate(entries):
        if i < start_idx:
            continue

        # If legal_basis already exists, keep it
        if entry.get("legal_basis", "").strip():
            processed.append(entry)
            continue

        label = entry.get("label", -1)
        violation_type = entry.get("violation_type", "none").strip().lower()
        text = entry.get("text", "").strip()

        # ── Compliant clause ──
        if label == 0:
            legal_basis = derive_compliant_legal_basis(text)
            rule_based_count += 1

        # ── Single known violation type ──
        elif violation_type in LEGAL_BASIS_MAP:
            legal_basis = LEGAL_BASIS_MAP[violation_type]
            rule_based_count += 1

        else:
            # ── Compound violation type (e.g. "dissolution_without_notice, exclusion_of_damages") ──
            parts = [p.strip() for p in violation_type.replace(";", ",").split(",")]
            known_parts = [p for p in parts if p in LEGAL_BASIS_MAP]
            unknown_parts = [p for p in parts if p not in LEGAL_BASIS_MAP]

            if known_parts:
                legal_basis = "; ".join(LEGAL_BASIS_MAP[p] for p in known_parts)
                rule_based_count += 1
            else:
                # Truly unknown — use generic fallback and log it
                legal_basis = (
                    "Article 148 — contract must be performed in good faith; "
                    "Article 150 — contract terms shall not be deviated from; "
                    f"violation type '{violation_type}' constitutes a breach of applicable Egyptian law provisions"
                )
                truly_unknown.append((i, violation_type))

            if unknown_parts:
                truly_unknown.append((i, ", ".join(unknown_parts)))

        # Update entry
        updated_entry = dict(entry)
        updated_entry["legal_basis"] = legal_basis
        processed.append(updated_entry)

        # Save progress periodically
        if (i + 1) % SAVE_EVERY == 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
            print(f"  [Checkpoint] Saved at entry {i+1}/{len(entries)}")

    # Final save
    print(f"\n[2] Saving output to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    # Clean up progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    # Summary
    print("\n" + "="*60)
    print("  COMPLETE")
    print("="*60)
    print(f"  Total entries:     {len(processed)}")
    print(f"  Rule-based:        {rule_based_count}")
    print(f"  Already had basis: {len(entries) - len(needs_basis)}")

    if truly_unknown:
        print(f"\n  [WARNING] {len(truly_unknown)} entries got generic fallback:")
        seen = set()
        for idx, vtype in truly_unknown:
            if vtype not in seen:
                print(f"    - Entry {idx}: '{vtype}'")
                seen.add(vtype)
        print("\n  Add these to LEGAL_BASIS_MAP and rerun to fix them.")
    else:
        print("\n  All entries matched — no fallbacks used.")

    print(f"\n  Output: {OUTPUT_FILE}")
    print("="*60 + "\n")

needs_basis = []  # initialized before main() so summary can reference it

if __name__ == "__main__":
    main()