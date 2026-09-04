use anyhow::{Result, anyhow};
use chronatello::{
    Candidate, CandidateKind, Classifier, Deadline, ModelOutput, ModelStatus, SCHEDULE_URL, State,
    calendar, extract_candidates, extract_candidates_with_overrides, reconcile, response_schema,
    validate_model,
};
use chrono::{TimeZone, Utc};
use std::{cell::RefCell, collections::HashMap, collections::VecDeque};

struct Fake(RefCell<VecDeque<Result<ModelOutput>>>);

impl Fake {
    fn outputs(outputs: Vec<ModelOutput>) -> Self {
        Self(RefCell::new(outputs.into_iter().map(Ok).collect()))
    }
}

impl Classifier for Fake {
    fn classify(&self, _: &Candidate) -> Result<ModelOutput> {
        self.0
            .borrow_mut()
            .pop_front()
            .unwrap_or_else(|| Err(anyhow!("unexpected classifier call")))
    }
}

fn schedule_html(row_date: &str, text: &str) -> String {
    format!(
        "<table><thead><tr><th>Week</th><th>Date</th><th>Topic</th><th>Prepare</th><th>Materials</th><th>In Class</th><th>Assignments</th></tr></thead><tbody><tr><td headers=\"week\">1</td><td headers=\"date\">{row_date}</td><td headers=\"topic\"></td><td headers=\"prepare\"></td><td headers=\"materials\"></td><td headers=\"in_class\"></td><td headers=\"assignments\">{text}</td></tr></tbody></table>"
    )
}

fn candidate_at(row_date: &str, text: &str) -> Candidate {
    extract_candidates(&schedule_html(row_date, text))
        .unwrap()
        .remove(0)
}

fn candidate(text: &str) -> Candidate {
    candidate_at("Tue 08/25", text)
}

fn all_day_output(evidence: &str, date: &str) -> ModelOutput {
    ModelOutput {
        status: ModelStatus::Event,
        title: "Homework 0".into(),
        body: "Complete the setup.".into(),
        classification: CandidateKind::Assignment,
        due_evidence: Some(evidence.into()),
        due_date: Some(date.into()),
        due_time: None,
        link_ids: vec![],
    }
}

fn no_deadline_output() -> ModelOutput {
    ModelOutput {
        status: ModelStatus::NoExplicitDeadline,
        title: String::new(),
        body: String::new(),
        classification: CandidateKind::Assignment,
        due_evidence: None,
        due_date: None,
        due_time: None,
        link_ids: vec![],
    }
}

#[test]
fn extracts_real_quarto_segments_and_resolves_links() {
    let candidates = extract_candidates(include_str!("fixtures/schedule.html")).unwrap();
    assert_eq!(candidates.len(), 4);
    assert_eq!(
        candidates
            .iter()
            .filter(|item| item.kind == CandidateKind::InClass)
            .count(),
        1
    );
    assert_eq!(candidates[0].links[0].text, "hw00");
    assert_eq!(
        candidates[0].links[0].url,
        "https://data385.netlify.app/hw/hw00-setup"
    );
    assert_eq!(
        candidates[2].links[0].url,
        "https://data385.netlify.app/hw/hw01-toolkit"
    );
    assert_eq!(candidates[3].source_text, "Finish d04");
}

#[test]
fn rejects_changed_table_schema() {
    let html = schedule_html("Tue 08/25", "Homework Due Thu 8/27")
        .replace("Assignments</th>", "Tasks</th>");
    assert!(extract_candidates(&html).is_err());
}

#[test]
fn validates_messy_exact_times_and_end_of_class() {
    let unrelated_time = candidate("Meet at 9 AM; report Due Thu 8/27");
    assert!(matches!(
        validate_model(
            &unrelated_time,
            all_day_output("Due Thu 8/27", "2026-08-27")
        )
        .unwrap(),
        Some(chronatello::StoredEvent {
            deadline: Deadline::AllDay { .. },
            ..
        })
    ));

    let timed = candidate("Project Due Tue 10/20 at 11:59 p.m.");
    let mut answer = all_day_output("Due Tue 10/20 at 11:59 p.m.", "2026-10-20");
    answer.due_time = Some("23:59".into());
    assert!(matches!(
        validate_model(&timed, answer).unwrap(),
        Some(chronatello::StoredEvent { deadline: Deadline::Timed { local }, .. })
            if local.to_string() == "2026-10-20 23:59:00"
    ));

    let class_html = schedule_html("Thu 08/27", "placeholder").replace(
        "<td headers=\"in_class\"></td><td headers=\"assignments\">placeholder</td>",
        "<td headers=\"in_class\">Lab Due end of class 8/27</td><td headers=\"assignments\"></td>",
    );
    let class = extract_candidates(&class_html).unwrap().remove(0);
    let mut missing_time = all_day_output("8/27", "2026-08-27");
    missing_time.classification = CandidateKind::InClass;
    assert!(validate_model(&class, missing_time).is_err());

    let mut answer = all_day_output("Due end of class 8/27", "2026-08-27");
    answer.classification = CandidateKind::InClass;
    answer.due_time = Some("end_of_class".into());
    assert!(matches!(
        validate_model(&class, answer).unwrap(),
        Some(chronatello::StoredEvent { deadline: Deadline::Timed { local }, .. })
            if local.to_string() == "2026-08-27 13:45:00"
    ));
}

#[test]
fn rejects_invalid_dates_weekdays_and_gemini_fields() {
    let item = candidate("Homework Due Mon 8/27");
    assert!(validate_model(&item, all_day_output("Due Mon 8/27", "2026-08-27")).is_err());

    let impossible = candidate("Homework Due Tue 9/31");
    assert!(validate_model(&impossible, all_day_output("Due Tue 9/31", "2026-09-30")).is_err());

    let mut invented = all_day_output("Due Mon 8/27", "2026-08-27");
    invented.link_ids.push("not_supplied".into());
    assert!(validate_model(&item, invented).is_err());
    let mut injected_url = all_day_output("Due Mon 8/27", "2026-08-27");
    injected_url.body = "Visit https://evil.invalid".into();
    assert!(validate_model(&item, injected_url).is_err());

    let ambiguous = candidate("Homework Due Thu 8/27 or Mon 8/31");
    assert!(validate_model(&ambiguous, all_day_output("Due Thu 8/27", "2026-08-27")).is_err());

    let explicit = candidate("Homework Due Thu 8/27");
    assert!(validate_model(&explicit, no_deadline_output()).is_err());
    let unrelated_date = candidate("Read notes from 8/27");
    assert!(validate_model(&unrelated_date, all_day_output("8/27", "2026-08-27")).is_err());
}

#[test]
fn uid_is_stable_when_deadline_and_row_change() {
    let now = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
    let first = candidate_at("Tue 08/25", "Homework Due Thu 8/27");
    let state = reconcile(
        State::empty(now),
        vec![first],
        &Fake::outputs(vec![all_day_output("Due Thu 8/27", "2026-08-27")]),
        now,
    )
    .unwrap();
    let uid = state.records[0].uid.clone();

    let later = now + chrono::Duration::days(1);
    let changed = candidate_at("Thu 09/03", "Homework Due Mon 8/31");
    let state = reconcile(
        state,
        vec![changed],
        &Fake::outputs(vec![all_day_output("Due Mon 8/31", "2026-08-31")]),
        later,
    )
    .unwrap();
    assert_eq!(state.records[0].uid, uid);
    assert_eq!(state.records[0].sequence, 1);
}

#[test]
fn explicit_code_and_manual_alias_keep_identity() {
    let coded_a = candidate_at("Tue 08/25", "hw01 Due Thu 8/27");
    let coded_b = candidate_at("Thu 09/03", "Complete Homework HW 01 by Mon 8/31");
    assert_eq!(coded_a.source_key, coded_b.source_key);

    let first_text = "Write the report Due Thu 8/27";
    let second_text = "Submit the final analysis by Mon 8/31";
    let first = extract_candidates(&schedule_html("Tue 08/25", first_text)).unwrap();
    let overrides = HashMap::from([(
        second_text.to_string(),
        format!("source_key:{}", first[0].source_key),
    )]);
    let renamed =
        extract_candidates_with_overrides(&schedule_html("Thu 09/03", second_text), &overrides)
            .unwrap();
    assert_eq!(first[0].source_key, renamed[0].source_key);
}

#[test]
fn state_retains_removals_and_reactivates_without_model_call() {
    let now = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
    let first = candidate("hw01 Due Thu 8/27");
    let second = candidate("hw02 Due Mon 8/31");
    let state = reconcile(
        State::empty(now),
        vec![first.clone(), second.clone()],
        &Fake::outputs(vec![
            all_day_output("Due Thu 8/27", "2026-08-27"),
            all_day_output("Due Mon 8/31", "2026-08-31"),
        ]),
        now,
    )
    .unwrap();
    let first_uid = state.records[0].uid.clone();

    let removed = reconcile(
        state,
        vec![second.clone()],
        &Fake::outputs(vec![]),
        now + chrono::Duration::hours(1),
    )
    .unwrap();
    assert!(!removed.records[0].active);
    assert_eq!(removed.records.len(), 2);

    let restored = reconcile(
        removed,
        vec![first, second],
        &Fake::outputs(vec![]),
        now + chrono::Duration::hours(2),
    )
    .unwrap();
    assert!(restored.records[0].active);
    assert_eq!(restored.records[0].uid, first_uid);
    assert_eq!(restored.records[0].sequence, 2);
}

#[test]
fn large_removal_is_rejected_even_for_two_active_events() {
    let now = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
    let first = candidate("hw01 Due Thu 8/27");
    let second = candidate("hw02 Due Mon 8/31");
    let state = reconcile(
        State::empty(now),
        vec![first, second],
        &Fake::outputs(vec![
            all_day_output("Due Thu 8/27", "2026-08-27"),
            all_day_output("Due Mon 8/31", "2026-08-31"),
        ]),
        now,
    )
    .unwrap();
    assert!(
        reconcile(
            state,
            vec![candidate("Finish d04")],
            &Fake::outputs(vec![no_deadline_output()]),
            now + chrono::Duration::hours(1),
        )
        .is_err()
    );
}

#[test]
fn new_item_without_deadline_is_cached_as_skipped() {
    let now = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
    let item = candidate("Finish d04");
    let state = reconcile(
        State::empty(now),
        vec![item],
        &Fake::outputs(vec![no_deadline_output()]),
        now,
    )
    .unwrap();
    assert!(!state.records[0].active);
    assert!(state.records[0].event.is_none());
}

#[test]
fn ics_has_links_crlf_unicode_folding_and_explicit_next_day_end() {
    let now = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
    let item = candidate(r#"<a href="/hw/hw00-setup">hw00</a> Due Thu 8/27"#);
    let mut answer = all_day_output("Due Thu 8/27", "2026-08-27");
    answer.body = "é".repeat(100);
    answer.link_ids = vec!["link_0".into()];
    let state = reconcile(
        State::empty(now),
        vec![item],
        &Fake::outputs(vec![answer]),
        now,
    )
    .unwrap();
    let ics = calendar(&state).unwrap();
    let _: icalendar::Calendar = ics.parse().unwrap();
    let unfolded = ics.replace("\r\n ", "");
    assert!(ics.contains("DTSTART;VALUE=DATE:20260827\r\n"));
    assert!(ics.contains("DTEND;VALUE=DATE:20260828\r\n"));
    assert!(ics.contains("URL:https://data385.netlify.app/hw/hw00-setup\r\n"));
    assert!(unfolded.contains("Original schedule text"));
    assert!(ics.contains("\r\n "));
    assert!(!ics.replace("\r\n", "").contains('\n'));
    for line in ics.split("\r\n") {
        assert!(line.len() <= 75, "overlong ICS line: {} bytes", line.len());
        assert!(std::str::from_utf8(line.as_bytes()).is_ok());
    }
    assert!(ics.contains(SCHEDULE_URL));
}

#[test]
fn timed_ics_uses_utc_without_an_undefined_timezone() {
    let now = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
    let item = candidate("Lab Due end of class 8/27");
    let mut answer = all_day_output("Due end of class 8/27", "2026-08-27");
    answer.due_time = Some("end_of_class".into());
    let state = reconcile(
        State::empty(now),
        vec![item],
        &Fake::outputs(vec![answer]),
        now,
    )
    .unwrap();
    let ics = calendar(&state).unwrap();
    assert!(ics.contains("DTSTART:20260827T204500Z\r\n"));
    assert!(!ics.contains("TZID="));
}

#[test]
fn strict_schema_and_deserialization_reject_extra_fields() {
    assert!(response_schema()["required"].is_array());
    let raw = r#"{"status":"event","title":"x","body":"y","classification":"assignment","due_evidence":null,"due_date":null,"due_time":null,"link_ids":[],"invented_url":"https://evil.invalid"}"#;
    assert!(serde_json::from_str::<ModelOutput>(raw).is_err());
}

#[test]
fn ambiguous_status_and_existing_event_losing_deadline_block() {
    let item = candidate("Homework Due Thu 8/27");
    let mut ambiguous = all_day_output("Due Thu 8/27", "2026-08-27");
    ambiguous.status = ModelStatus::Ambiguous;
    assert!(validate_model(&item, ambiguous).is_err());

    let now = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
    let state = reconcile(
        State::empty(now),
        vec![item],
        &Fake::outputs(vec![all_day_output("Due Thu 8/27", "2026-08-27")]),
        now,
    )
    .unwrap();
    let changed = candidate("Homework Due date TBD");
    assert!(
        reconcile(
            state,
            vec![changed],
            &Fake::outputs(vec![no_deadline_output()]),
            now
        )
        .is_err()
    );
}
